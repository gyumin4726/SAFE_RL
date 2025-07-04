import os
import gym
import sys
import yaml
import time
import torch
import warnings
import numpy as np
from tqdm import tqdm
from itertools import count
from collections import deque
import matplotlib.pyplot as plt

sys.path.append('/home/kang/code/rndix/safehil-llm/SMARTS')
from smarts.core.agent import AgentSpec
from smarts.env.hiway_env import HiWayEnv
from smarts.core.controllers import ActionSpaceType
from smarts.core.agent_interface import AgentInterface
from smarts.core.agent_interface import NeighborhoodVehicles, RGB, OGM, DrivableAreaGridMap, Waypoints

from stable_baselines3.common.vec_env import DummyVecEnv  # Import DummyVecEnv

sys.path.append('/home/20201914/safehil-llm/Auto_Driving_Highway')
from scenario import Scenario
from customTools import (
    getAvailableActions,
    getAvailableLanes,
    getLaneInvolvedCar,
    isChangeLaneConflictWithCar,
    isAccelerationConflictWithCar,
    isKeepSpeedConflictWithCar,
    isDecelerationSafe,
    isActionSafe
)
from analysis_obs import available_action, get_available_lanes, get_involved_cars, extract_lanes_info, extract_lane_and_car_ids, assess_lane_change_safety, check_safety_in_current_lane, format_training_info
import ask_llm as ask_llm
from ask_llm import ACTIONS_ALL

from drl_agent import DRL
from keyboard import HumanKeyboardAgent
from utils_ import soft_update, hard_update
from authority_allocation import Arbitrator
from main import observation_adapter, action_adapter, reward_adapter

class MyHiWayEnv(gym.Env):
    def __init__(self, screen_size, scenario_path, agent_specs, seed, vehicleCount,
                 observation_space, action_space, agent_id,
                 headless=False, visdom=False, sumo_headless=True):
        super(MyHiWayEnv, self).__init__()
        
        self.screen_size = screen_size
        self.states = np.zeros(shape=(screen_size, screen_size, 9), dtype=np.float32)
        self.meta_state = None

        self.scenario_path = scenario_path
        self.agent_specs = agent_specs
        self.seed = seed
        self.vehicleCount = vehicleCount

        self.observation_space = observation_space
        self.action_space = action_space
        self.agent_id = agent_id
        self.headless = headless
        self.visdom = visdom
        self.sumo_headless = sumo_headless

        self.env = HiWayEnv(
            scenarios=self.scenario_path,
            agent_specs=self.agent_specs,
            headless=self.headless,
            visdom=self.visdom,
            sumo_headless=self.sumo_headless,
            seed=self.seed
        )

    def step(self, action):
        # Step the wrapped environment and capture all returned values
        meta_state, reward, done, truncated = self.env.step(action)
        self.meta_state = meta_state[self.agent_id]
        obs = self.observation_adapter(self.meta_state)
        self.last_observation = obs
        custom_reward = self.calculate_custom_reward(tuple(action.values())[0])

        return obs, custom_reward, done, truncated

    def set_llm_suggested_action(self, action):
        self.llm_suggested_action = action
    
    def calculate_custom_reward(self, action):
        action_name, _ = ask_llm.get_action_info(action)
        llm_action = self.llm_suggested_action
    
        # 정확히 일치하는 경우
        if action_name in llm_action:
            reward = 1.0
            print(f"✅ 액션 일치! 보상: {reward}")
            return reward
    
        # 부분적으로 유사한 경우 (속도 관련 액션)
        speed_actions = ["FASTER", "SLOWER"]
        if (action_name in speed_actions and llm_action in speed_actions):
            reward = 0.3
            print(f"⚠️ 액션 부분 일치! 보상: {reward}")
            return reward

        if hasattr(self, '_debug_step') and self._debug_step:
            if action_name in llm_action:
                print(f"✅ 액션 일치! 보상: {reward}")
        
        # 일치하지 않는 경우
        print(f"❌ 액션 불일치! 보상: 0")
        return 0.0
    
    def observation_adapter(self, meta_state):
        new_obs = meta_state.top_down_rgb[1] / 255.0
        self.states[:, :, 0:3] = self.states[:, :, 3:6]
        self.states[:, :, 3:6] = self.states[:, :, 6:9]
        self.states[:, :, 6:9] = new_obs
        ogm = meta_state.occupancy_grid_map[1] 
        drivable_area = meta_state.drivable_area_grid_map[1]

        if meta_state.events.collisions or meta_state.events.reached_goal:
            self.states = np.zeros(shape=(self.screen_size, self.screen_size, 9), dtype=np.float32)

        return self.states
    
    def reset(self, **kwargs):
        meta_state = self.env.reset(**kwargs)
        self.meta_state = meta_state[self.agent_id]
        state = self.observation_adapter(self.meta_state)

        self.last_observation = state
        return state
    
    def get_available_actions(self):
        """Get the list of available actions from the underlying Highway environment."""
        sce = Scenario(vehicleCount=self.vehicleCount)
        sce.updateVehicles(self.meta_state, 0)

        toolModels = [
            getAvailableActions(self.env.unwrapped),
            getAvailableLanes(sce),
            getLaneInvolvedCar(sce),
            isChangeLaneConflictWithCar(sce),
            isAccelerationConflictWithCar(sce),
            isKeepSpeedConflictWithCar(sce),
            isDecelerationSafe(sce),
    ]

        available = available_action(toolModels)
        valid_action_ids = [i for i, act in ACTIONS_ALL.items() if available.get(act, False)]
        return valid_action_ids

def train(env, agent, sce, toolModels):
    save_threshold = 3.0
    trigger_reward = 3.0
    trigger_epoc = 400
    saved_epoc = 1
    epoc = 0
    pbar = tqdm(total=MAX_NUM_EPOC)
    frame = 0
    
    # Initialize reward tracking lists
    reward_list = []
    reward_mean_list = []
    
    while epoc <= MAX_NUM_EPOC:
        reward_total = 0.0 
        error = 0.0 
        action_list = deque(maxlen=1)
        action_list.append(np.array([0.0, 0.0]))
        guidance_count = int(0)
        guidance_rate = 0.0
        frame_skip = 5
        
        continuous_threshold = 100
        intermittent_threshold = 300
        
        pos_list = deque(maxlen=5)
        s = env.reset()
        obs = env.envs[0].meta_state
        initial_pos = obs.ego_vehicle_state.position[:2]
        pos_list.append(initial_pos)

        for t in count():
            if t > MAX_NUM_STEPS:
                print('Max Steps Done.')
                break
            
            sce.updateVehicles(obs, frame)
            # Observation translation
            msg0 = available_action(toolModels)
            msg1 = get_available_lanes(toolModels)
            msg2 = get_involved_cars((toolModels))
            msg1_info = next(iter(msg1.values()))
            lanes_info = extract_lanes_info(msg1_info)

            lane_car_ids = extract_lane_and_car_ids(lanes_info, msg2)
            safety_assessment = assess_lane_change_safety(toolModels, lane_car_ids)
            lane_change_safety = assess_lane_change_safety(toolModels, lane_car_ids)
            safety_msg = check_safety_in_current_lane(toolModels, lane_car_ids)
            formatted_info = format_training_info(msg0, msg1, msg2, lanes_info, lane_car_ids, safety_assessment, lane_change_safety, safety_msg)

            ##### Select and perform an action ######
            rl_a = agent.choose_action(s[0], action_list[-1])
            guidance = False

            ##### Human-in-the-loop #######
            if model != 'SAC' and human.intervention and epoc <= INTERMITTENT_THRESHOLD:
                human_action = human.act()
                guidance = True
            else:
                human_a = np.array([0.0, 0.0])
            
            ###### Assign final action ######
            if guidance:
                if human_action[1] > human.MIN_BRAKE:
                    human_a = np.array([-human_action[1], human_action[-1]])
                else:
                    human_a = np.array([human_action[0], human_action[-1]])
                
                if arbitrator.shared_control and epoc > CONTINUAL_THRESHOLD:
                    rl_authority, human_authority = arbitrator.authority(obs, rl_a, human_a)
                    a = rl_authority * rl_a + human_authority * human_a
                else:
                    a = human_a
                    human_authority = 1.0 #np.array([1.0, 1.0])
                engage = int(1)
                authority = human_authority
                guidance_count += int(1)
            else:
                a = rl_a
                engage = int(0)
                authority = 0.0 

            ##### Interaction #####
            action = action_adapter(a)
            
            ##### Safety Mask #####
            ego_state = obs.ego_vehicle_state
            lane_id = ego_state.lane_index
            if ego_state.speed >= obs.waypoint_paths[lane_id][0].speed_limit and\
               action[0] > 0.0:
                   action = list(action)
                   action[0] = 0.0
                   action = tuple(action)

            llm_response = ask_llm.send_to_chatgpt(action, formatted_info, sce)
            decision_content = llm_response.content
            print(llm_response)
            llm_suggested_action = extract_decision(decision_content)
            # llm_suggested_action = 'FASTER'  # 하드코딩 비활성화
            print(f"llm action: {llm_suggested_action}")

            env.env_method('set_llm_suggested_action', llm_suggested_action)
            # print(f"Action: {action}")
            # print(f"Observation: {next_obs}")

            action = {AGENT_ID:action}

            s_, custom_reward, done, info = env.step([action])
            custom_reward = custom_reward[0]
            done = done[0]
            info = info[0]
            print(f"Reward: {custom_reward}\n")

            obs = env.envs[0].meta_state
            curr_pos = obs.ego_vehicle_state.position[:2]
            
            if env_name == 'straight' and (curr_pos[0] - initial_pos[0]) > 200:
                done = True
                print('Done')
            elif env_name == 'straight_with_turn' and (curr_pos[1] - initial_pos[1]) > 98:
                done = True
                print('Done')
            
            r = reward_adapter(obs, pos_list, a, engage=engage, done=done)
            pos_list.append(curr_pos)

            ##### Store the transition in memory ######
            agent.store_transition(s, action_list[-1], a, human_a, r,
                                   s_, a, engage, authority, done)
            
            reward_total += r + custom_reward
            action_list.append(a)
            s = s_
            frame += 1
                            
            if epoc >= THRESHOLD:   
                # Train the DRL model
                agent.learn_guidence(BATCH_SIZE)

            if human.slow_down:
                time.sleep(1/40)

            if done:
                epoc += 1
                if epoc > THRESHOLD:
                    reward_list.append(max(-15.0, reward_total))
                    reward_mean_list.append(np.mean(reward_list[-10:]))
                
                    ###### Evaluating the performance of current model ######
                    if reward_mean_list[-1] >= trigger_reward and epoc > trigger_epoc:
                        # trigger_reward = reward_mean_list[-1]
                        print("Evaluating the Performance.")
                        avg_reward, _, _, _, _ = evaluate(env, agent, EVALUATION_EPOC)
                        trigger_reward = avg_reward
                        if avg_reward > save_threshold:
                            print('Save the model at %i epoch, reward is: %f' % (epoc, avg_reward))
                            saved_epoc = epoc
                            
                            torch.save(agent.policy.state_dict(), os.path.join('trained_network/' + env_name,
                                      name+'_memo'+str(MEMORY_CAPACITY)+'_epoc'+
                                      str(MAX_NUM_EPOC) + '_step' + str(MAX_NUM_STEPS) + '_seed'
                                      + str(seed)+'_'+env_name+'_actornet.pkl'))
                            torch.save(agent.critic.state_dict(), os.path.join('trained_network/' + env_name,
                                      name+'_memo'+str(MEMORY_CAPACITY)+'_epoc'+
                                      str(MAX_NUM_EPOC) + '_step' + str(MAX_NUM_STEPS) + '_seed'
                                      + str(seed)+'_'+env_name+'_criticnet.pkl'))
                            save_threshold = avg_reward

                print('\n|Epoc:', epoc,
                      '\n|Step:', t,
                      '\n|Goal:', info[AGENT_ID]['env_obs'].events.reached_goal,
                      '\n|Guidance Rate:', guidance_rate, '%',
                      '\n|Collision:', bool(len(info[AGENT_ID]['env_obs'].events.collisions)),
                      '\n|Off Road:', info[AGENT_ID]['env_obs'].events.off_road,
                      '\n|Off Route:', info[AGENT_ID]['env_obs'].events.off_route,
                      '\n|R:', reward_total,
                      '\n|Temperature:', agent.alpha,
                      '\n|Reward Threshold:', save_threshold,
                      '\n|Algo:', name,
                      '\n|seed:', seed,
                      '\n|Env:', env_name)
    
                s = env.reset()
                reward_total = 0
                error = 0
                pbar.update(1)
                break
        
        # if epoc % PLOT_INTERVAL == 0:
        #     plot_animation_figure(saved_epoc)
    
        if (epoc % SAVE_INTERVAL == 0):
            np.save(os.path.join('store/' + env_name, 'reward_memo'+str(MEMORY_CAPACITY) +
                                      '_epoc'+str(MAX_NUM_EPOC)+'_step' + str(MAX_NUM_STEPS) +
                                      '_seed'+ str(seed) +'_'+env_name+'_' + name),
                    [reward_mean_list], allow_pickle=True, fix_imports=True)

    pbar.close()
    print('Complete')
    
    # Save final model after all epochs are completed
    print(f'학습 완료! 최종 모델 저장 중... (에폭: {epoc})')
    torch.save(agent.policy.state_dict(), os.path.join('trained_network/' + env_name,
              name+'_final_memo'+str(MEMORY_CAPACITY)+'_epoc'+
              str(MAX_NUM_EPOC) + '_step' + str(MAX_NUM_STEPS) + '_seed'
              + str(seed)+'_'+env_name+'_actornet.pkl'))
    torch.save(agent.critic.state_dict(), os.path.join('trained_network/' + env_name,
              name+'_final_memo'+str(MEMORY_CAPACITY)+'_epoc'+
              str(MAX_NUM_EPOC) + '_step' + str(MAX_NUM_STEPS) + '_seed'
              + str(seed)+'_'+env_name+'_criticnet.pkl'))
    
    print('최종 모델 저장 완료! (Actor + Critic 네트워크)')
    return save_threshold

# utils.py
def extract_decision(response_content):
    try:
        import re
        pattern = r'"decision":\s*{\s*"([^"]+)"\s*}'
        match = re.search(pattern, response_content)
        if match:
            raw_decision = match.group(1).upper().strip()
            
            # 매핑 사전
            decision_map = {
                "ACCELERATE": "FASTER",
                "SPEED UP": "FASTER",
                "GO FASTER": "FASTER",
                "DECELERATE": "SLOWER",
                "SLOW DOWN": "SLOWER",
                "BRAKE": "SLOWER",
                "STAY": "IDLE",
                "MAINTAIN": "IDLE",
                "KEEP": "IDLE",
                "LEFT": "LANE_LEFT",
                "CHANGE LEFT": "LANE_LEFT",
                "RIGHT": "LANE_RIGHT",
                "CHANGE RIGHT": "LANE_RIGHT"
            }
            
            # 매핑 시도
            if raw_decision in {'FASTER', 'SLOWER', 'LEFT', 'IDLE', 'RIGHT'}:
                return raw_decision  # 이미 올바른 형식
            else:
                mapped = decision_map.get(raw_decision)
                print(f"LLM 응답 '{raw_decision}'을(를) '{mapped}'(으)로 매핑")
                return mapped
        
        print(f"결정 패턴을 찾을 수 없음: {response_content}")
        return None
    except Exception as e:
        print(f"결정 추출 중 오류: {e}")
        return None




if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

    plt.ion()
    
    path = os.getcwd()
    yaml_path = os.path.join(path, 'config.yaml')
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    ##### Individual parameters for each model ######
    model = 'SAC'
    mode_param = config[model]
    name = mode_param['name']
    POLICY_GUIDANCE = mode_param['POLICY_GUIDANCE']
    VALUE_GUIDANCE = mode_param['VALUE_GUIDANCE']
    PENALTY_GUIDANCE = mode_param['PENALTY_GUIDANCE']
    ADAPTIVE_CONFIDENCE = mode_param['ADAPTIVE_CONFIDENCE']
    
    if model != 'SAC':
        SHARED_CONTROL = mode_param['SHARED_CONTROL']
        CONTINUAL_THRESHOLD = mode_param['CONTINUAL_THRESHOLD']
        INTERMITTENT_THRESHOLD = mode_param['INTERMITTENT_THRESHOLD']
    else:
        SHARED_CONTROL = False
        
    ###### Default parameters for DRL ######
    mode = config['mode']
    ACTOR_TYPE = config['ACTOR_TYPE']
    CRITIC_TYPE = config['CRITIC_TYPE']
    LR_ACTOR = config['LR_ACTOR']
    LR_CRITIC = config['LR_CRITIC']
    TAU = config['TAU']
    THRESHOLD = config['THRESHOLD']
    TARGET_UPDATE = config['TARGET_UPDATE']
    BATCH_SIZE = config['BATCH_SIZE']
    GAMMA = config['GAMMA']
    MEMORY_CAPACITY = config['MEMORY_CAPACITY']
    MAX_NUM_EPOC = config['MAX_NUM_EPOC']
    MAX_NUM_STEPS = config['MAX_NUM_STEPS']
    PLOT_INTERVAL = config['PLOT_INTERVAL']
    SAVE_INTERVAL = config['SAVE_INTERVAL']
    EVALUATION_EPOC = config['EVALUATION_EPOC']

    ###### Entropy ######
    ENTROPY = config['ENTROPY']
    LR_ALPHA = config['LR_ALPHA']
    ALPHA = config['ALPHA']
    
    ###### Env Settings #######
    env_name = config['env_name']
    scenario = config['scenario_path']
    screen_size = config['screen_size']
    view = config['view']
    condition_state_dim = config['condition_state_dim']
    AGENT_ID = config['AGENT_ID']
    
    # Create the network storage folders
    if not os.path.exists("./store/" + env_name):
        os.makedirs("./store/" + env_name)
        
    if not os.path.exists("./trained_network/" + env_name):
        os.makedirs("./trained_network/" + env_name)

    ##### Train #####
    for i in range(1, 2):
        seed = i

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        #### Environment specs ####
        ACTION_SPACE = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
        OBSERVATION_SPACE = gym.spaces.Box(low=0, high=1, shape=(screen_size, screen_size, 9))
        states = np.zeros(shape=(screen_size, screen_size, 9), dtype=np.float32)
    
        ##### Define agent interface #######
        agent_interface = AgentInterface(
            max_episode_steps=MAX_NUM_STEPS,
            waypoints=Waypoints(50),
            neighborhood_vehicles=NeighborhoodVehicles(radius=100),
            rgb=RGB(screen_size, screen_size, view/screen_size),
            ogm=OGM(screen_size, screen_size, view/screen_size),
            drivable_area_grid_map=DrivableAreaGridMap(screen_size, screen_size, view/screen_size),
            action=ActionSpaceType.Continuous,
        )
        
        ###### Define agent specs ######
        agent_spec = AgentSpec(
            interface=agent_interface,
            # observation_adapter=observation_adapter,
            # reward_adapter=reward_adapter,
            # action_adapter=action_adapter,
            # info_adapter=info_adapter,
        )
        
        ######## Human Intervention through g29 or keyboard ########
        human = HumanKeyboardAgent()
        
        ##### Create Env ######
        if model == 'SAC':
            envisionless = True
        else:
            envisionless = False
        
        scenario_path = [scenario]
        env = MyHiWayEnv(screen_size, scenario_path, agent_specs={AGENT_ID: agent_spec},
                       headless=False, visdom=False, sumo_headless=True, seed=seed, vehicleCount=5,
                       observation_space = OBSERVATION_SPACE, action_space = ACTION_SPACE, agent_id = AGENT_ID)
        env = DummyVecEnv([lambda: env])

        obs = env.reset()
        img_h, img_w, channel = screen_size, screen_size, 9
        physical_state_dim = 2
        n_obs = img_h * img_w * channel
        n_actions = env.action_space.high.size
        
        legend_bar = []
        
        # Initialize the agent
        agent = DRL(seed, n_actions, channel, condition_state_dim, ACTOR_TYPE, CRITIC_TYPE,
                    LR_ACTOR, LR_CRITIC, LR_ALPHA, MEMORY_CAPACITY, TAU,
                    GAMMA, ALPHA, POLICY_GUIDANCE, VALUE_GUIDANCE,
                    ADAPTIVE_CONFIDENCE, ENTROPY)
        
        arbitrator = Arbitrator()
        arbitrator.shared_control = SHARED_CONTROL

        sce = Scenario(vehicleCount=5)
        toolModels = [
            getAvailableActions(env.envs[0]),
            getAvailableLanes(sce),
            getLaneInvolvedCar(sce),
            isChangeLaneConflictWithCar(sce),
            isAccelerationConflictWithCar(sce),
            isKeepSpeedConflictWithCar(sce),
            isDecelerationSafe(sce),
            # isActionSafe()
        ]
    
        train(env, agent, sce, toolModels)
        
        legend_bar.append('seed'+str(seed))
        
        train_durations = []
        train_durations_mean_list = []
        reward_list = []
        reward_mean_list = []
        guidance_list = []

        
        print('\nThe object is:', model, '\n|Seed:', agent.seed, 
             '\n|VALUE_GUIDANCE:', VALUE_GUIDANCE, '\n|PENALTY_GUIDANCE:', PENALTY_GUIDANCE,'\n')
        
        success_count = 0

        if mode == 'evaluation':
            name = 'sac'
            max_epoc = 820
            max_steps = 300
            seed = 4
            directory = 'trained_network/' + env_name
            # directory =  'best_candidate'
            filename = name+'_memo'+str(MEMORY_CAPACITY)+'_epoc'+ \
                      str(max_epoc) + '_step' + str(max_steps) + \
                      '_seed' + str(seed) + '_' + env_name
                      
            agent.policy.load_state_dict(torch.load('%s/%s_actornet.pkl' % (directory, filename)))
            agent.policy.eval()
            reward, v_list_avg, offset_list_avg, dist_list_avg, avg_reward_list = evaluate(env, agent, eval_episodes=10)
            
            print('\n|Avg Speed:', np.mean(v_list_avg),
                  '\n|Std Speed:', np.std(v_list_avg),
                  '\n|Avg Dist:', np.mean(dist_list_avg),
                  '\n|Std Dist:', np.std(dist_list_avg),
                  '\n|Avg Offset:', np.mean(offset_list_avg),
                  '\n|Std Offset:', np.std(offset_list_avg))

        else:
            save_threshold = train(success_count)
        
            np.save(os.path.join('store/' + env_name, 'reward_memo'+str(MEMORY_CAPACITY) +
                                      '_epoc'+str(MAX_NUM_EPOC)+'_step' + str(MAX_NUM_STEPS) +
                                      '_seed'+ str(seed) +'_'+env_name+'_' + name),
                    [reward_mean_list], allow_pickle=True, fix_imports=True)
    
            torch.save(agent.policy.state_dict(), os.path.join('trained_network/' + env_name,
                      name+'_memo'+str(MEMORY_CAPACITY)+'_epoc'+
                      str(MAX_NUM_EPOC) + '_step' + str(MAX_NUM_STEPS) + '_seed'
                      + str(seed)+'_'+env_name+'_actornet_final.pkl'))
            torch.save(agent.critic.state_dict(), os.path.join('trained_network/' + env_name,
                      name+'_memo'+str(MEMORY_CAPACITY)+'_epoc'+
                      str(MAX_NUM_EPOC) + '_step' + str(MAX_NUM_STEPS) + '_seed'
                      + str(seed)+'_'+env_name+'_criticnet_final.pkl'))

        env.close()