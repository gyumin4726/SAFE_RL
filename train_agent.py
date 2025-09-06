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
from datetime import datetime
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), 'SMARTS'))
from smarts.core.agent import AgentSpec
from smarts.env.gymnasium.hiway_env_v1 import HiWayEnvV1 as HiWayEnv
from smarts.core.controllers import ActionSpaceType
from smarts.core.agent_interface import AgentInterface
from smarts.core.agent_interface import NeighborhoodVehicles, RGB, OGM, DrivableAreaGridMap, Waypoints

from stable_baselines3.common.vec_env import DummyVecEnv  # Import DummyVecEnv

sys.path.append(os.path.join(os.path.dirname(__file__), 'Auto_Driving_Highway'))
sys.path.append('Auto_Driving_Highway')
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
from main import observation_adapter, action_adapter, reward_adapter, evaluate  # evaluate 함수 추가

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
AGENT_ID = 'AGENT-007'  # Default agent ID
if 'AGENT_ID' in config:
    AGENT_ID = config['AGENT_ID']

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
        def normalize_action(val):
            if isinstance(val, (set, list, tuple)):
                val = list(val)[0] if val else ""
            if isinstance(val, dict):
                val = list(val.keys())[0] if val else ""
            return str(val).strip().upper()

        # get_action_info에 전달되는 action 벡터 출력
        print(f"[DEBUG] get_action_info input action: {action} ({type(action)})")

        action_name, _ = ask_llm.get_action_info(action)
        llm_action = self.llm_suggested_action

        # 디버깅용 출력
        print(f"[DEBUG] action_name: {action_name} ({type(action_name)})")
        print(f"[DEBUG] llm_action: {llm_action} ({type(llm_action)})")
        print(f"[DEBUG] normalize_action(action_name): {normalize_action(action_name)}")
        print(f"[DEBUG] normalize_action(llm_action): {normalize_action(llm_action)}")

        # robust 비교
        if normalize_action(action_name) == normalize_action(llm_action):
            reward = 1.0
            print(f"✅ 액션 일치! 보상: {reward}")
            return reward

        # 부분적으로 유사한 경우 (속도 관련 액션)
        speed_actions = ["FASTER", "SLOWER"]
        if (action_name in speed_actions and llm_action in speed_actions):
            reward = 0.3
            print(f"⚠️ 액션 부분 일치! 보상: {reward}")
            return reward

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

def train(env, agent, sce, toolModels, start_epoch=0):
    save_threshold = 3.0
    trigger_reward = 3.0
    trigger_epoc = 400
    saved_epoc = 1
    epoc = start_epoch
    pbar = tqdm(total=MAX_NUM_EPOC)
    frame = 0
    
    # 날짜별 저장 폴더 생성
    date_str = datetime.now().strftime('%Y%m%d')
    save_dir = os.path.join('trained_network', env_name, date_str)
    os.makedirs(save_dir, exist_ok=True)
    
    # 체크포인트 파일 경로
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{start_epoch}.pt')
    
    # 이전 체크포인트에서 reward 리스트 불러오기
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        reward_list = checkpoint.get('reward_list', [])
        reward_mean_list = checkpoint.get('reward_mean_list', [])
    else:
        reward_list = []
        reward_mean_list = []

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
            print(f"epoc: {epoc}, INTERMITTENT_THRESHOLD: {INTERMITTENT_THRESHOLD}")
            human_action = get_human_action()
            print(f"human_action: {human_action}")
            if human_action:
                # 키보드 입력과 동일하게 액션 벡터로 변환
                ACTION_VECTOR_MAP = {
                    "LANE_LEFT":  [0.0, 0.0, -1.0],
                    "IDLE":       [0.15, 0.0, 0.0],
                    "LANE_RIGHT": [0.0, 0.0, 1.0],
                    "FASTER":     [0.6, 0.0, 0.0],
                    "SLOWER":     [0.0, 1.0, 0.0],
                }
                action_vec = ACTION_VECTOR_MAP.get(human_action, [0.0, 0.0, 0.0])
                guidance = True
                human_a = np.array(action_vec)
                print(f"입력된 액션 '{human_action}'이(가) 반영되었습니다. (벡터: {action_vec})")
            elif human.intervention:
                # 기존 키보드 입력도 지원
                # (여기에 기존 human.act() 코드 또는 키보드 입력 반영 코드 삽입)
                human_a = np.array([human.throttle, human.steering_angle])
                guidance = True
            else:
                human_a = np.array([0.0, 0.0])
            
            ###### Assign final action ######
            if guidance:
                if human_a[1] > human.MIN_BRAKE:
                    human_a = np.array([-human_a[1], human_a[-1]])
                else:
                    human_a = np.array([human_a[0], human_a[-1]])
                
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
            print(f"[DEBUG] before action_adapter: a = {a} ({type(a)})")
            action = action_adapter(a)
            print(f"[DEBUG] after action_adapter: action = {action} ({type(action)})")
            
            ##### Safety Mask #####
            ego_state = obs.ego_vehicle_state
            lane_id = ego_state.lane_index
            # if ego_state.speed >= obs.waypoint_paths[lane_id][0].speed_limit and\
            #    action[0] > 0.0:
            #        action = list(action)
            #        action[0] = 0.0
            #        action = tuple(action)

            llm_response = ask_llm.send_to_chatgpt(action, formatted_info, sce)
            decision_content = llm_response
            print(llm_response)
            llm_suggested_action = extract_decision(decision_content)
            # llm_suggested_action = 'FASTER'  # 하드코딩 비활성화
            print(f"llm action: {llm_suggested_action}")

            env.env_method('set_llm_suggested_action', llm_suggested_action)
            # print(f"Action: {action}")
            # print(f"Observation: {next_obs}")

            action = {AGENT_ID:action}
            print(f"[DEBUG] env.step에 전달되는 action: {action}")

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
            
            r = reward_adapter(obs, pos_list, a, engage=engage, done=done, PENALTY_GUIDANCE=PENALTY_GUIDANCE)
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
                
                    # 5 에폭마다 체크포인트 저장
                    if epoc % 5 == 0:
                        checkpoint = {
                            'epoch': epoc,
                            'actor_state_dict': agent.policy.state_dict(),
                            'critic_state_dict': agent.critic.state_dict(),
                            'reward_list': reward_list,
                            'reward_mean_list': reward_mean_list
                        }
                        # 새로운 체크포인트 파일 경로
                        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoc}.pt')
                        torch.save(checkpoint, checkpoint_path)
                        print(f'\n체크포인트 저장 완료 (에폭 {epoc})')

                    ###### Evaluating the performance of current model ######
                    if reward_mean_list[-1] >= trigger_reward and epoc > trigger_epoc:
                        print("Evaluating the Performance.")
                        # DummyVecEnv에서 실제 환경 가져오기
                        real_env = env.envs[0] if hasattr(env, 'envs') else env
                        avg_reward, _, _, _, _ = evaluate(real_env.env, agent, EVALUATION_EPOC, agent_id=AGENT_ID,
                                                        observation_adapter=real_env.observation_adapter,
                                                        max_steps=MAX_NUM_STEPS,
                                                        env_name=config['env_name'],
                                                        human=human,
                                                        name=name,
                                                        seed=seed)  # seed 변수 전달
                        trigger_reward = avg_reward
                        if avg_reward > save_threshold:
                            print('Save the model at %i epoch, reward is: %f' % (epoc, avg_reward))
                            saved_epoc = epoc
                            
                            # 모델 저장 경로 수정
                            actor_path = os.path.join(save_dir, f'actor_epoch_{epoc}.pkl')
                            critic_path = os.path.join(save_dir, f'critic_epoch_{epoc}.pkl')
                            
                            torch.save(agent.policy.state_dict(), actor_path)
                            torch.save(agent.critic.state_dict(), critic_path)
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
    
    # 최종 모델 저장
    final_actor_path = os.path.join(save_dir, 'actor_final.pkl')
    final_critic_path = os.path.join(save_dir, 'critic_final.pkl')
    torch.save(agent.policy.state_dict(), final_actor_path)
    torch.save(agent.critic.state_dict(), final_critic_path)
    print(f'\n최종 모델 저장 완료 (에폭 {epoc})')
    
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
            if raw_decision in {'FASTER', 'SLOWER', 'LANE_LEFT', 'IDLE', 'LANE_RIGHT'}:
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

def find_latest_checkpoint(env_name):
    """가장 최근 체크포인트를 찾는 함수"""
    base_dir = os.path.join('trained_network', env_name)
    if not os.path.exists(base_dir):
        return None, 0
    
    # 날짜 폴더 찾기
    date_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not date_dirs:
        return None, 0
    
    latest_date = sorted(date_dirs)[-1]  # 가장 최근 날짜
    date_dir = os.path.join(base_dir, latest_date)
    
    # 체크포인트 파일 찾기
    checkpoints = [f for f in os.listdir(date_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
    if not checkpoints:
        return None, 0
    
    # 가장 높은 에폭의 체크포인트 찾기
    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))[-1]
    epoch = int(latest_checkpoint.split('_epoch_')[1].split('.')[0])
    
    return os.path.join(date_dir, latest_checkpoint), epoch

def get_human_action():
    print("get_human_action() 호출됨")
    try:
        with open("human_action.txt", "r") as f:
            action = f.read().strip()
        os.remove("human_action.txt")
        print(f"읽은 액션: {action}")
        return action
    except FileNotFoundError:
        return None


if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

    # Argument parser 추가
    parser = argparse.ArgumentParser(description='Train or evaluate agent')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluation'], 
                       help='Mode: train or evaluation')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file for evaluation')
    args = parser.parse_args()

    plt.ion()
    
    path = os.getcwd()
    yaml_path = os.path.join(path, 'config.yaml')
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 명령줄 인자로 받은 mode를 config에 적용
    if args.mode:
        config['mode'] = args.mode
    
    ##### Individual parameters for each model ######
    model = 'SaHiL'
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
        
        # 평가 모드가 아닐 때만 초기 체크포인트 로딩
        if mode != 'evaluation':
            # 가장 최근 체크포인트 찾기
            latest_checkpoint_path, start_epoch = find_latest_checkpoint(env_name)
            if latest_checkpoint_path:
                print(f"\n체크포인트를 불러오는 중... ({latest_checkpoint_path})")
                checkpoint = torch.load(latest_checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
                agent.policy.load_state_dict(checkpoint['actor_state_dict'])
                agent.critic.load_state_dict(checkpoint['critic_state_dict'])
                print(f"체크포인트 불러오기 완료 (에폭 {start_epoch}부터 시작)")
            else:
                start_epoch = 0
                print("체크포인트가 없습니다. 처음부터 학습을 시작합니다.")
        else:
            start_epoch = 0  # 평가 모드에서는 start_epoch를 0으로 설정
        
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
        ]
    
        # 평가 모드가 아닐 때만 학습 실행
        if mode != 'evaluation':
            train(env, agent, sce, toolModels, start_epoch)
        
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
                      
            # 명령줄 인자로 받은 체크포인트 경로 사용
            if args.checkpoint:
                checkpoint_path = args.checkpoint
                print(f"명령줄 인자로 받은 체크포인트를 불러옵니다: {checkpoint_path}")
                if os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
                    agent.policy.load_state_dict(checkpoint['actor_state_dict'])
                    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
                    print(f"체크포인트 불러오기 완료: {checkpoint_path}")
                else:
                    print(f"체크포인트 파일이 존재하지 않습니다: {checkpoint_path}")
                    sys.exit(1)
            else:
                checkpoint_path = os.path.join(directory, filename)
                if os.path.exists(checkpoint_path):
                    print(f"기본 체크포인트를 불러옵니다: {checkpoint_path}")
                    agent.policy.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
                else:
                    latest_checkpoint_path, start_epoch = find_latest_checkpoint(env_name)
                    if latest_checkpoint_path:
                        print(f"\n체크포인트를 불러오는 중... ({latest_checkpoint_path})")
                        checkpoint = torch.load(latest_checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
                        agent.policy.load_state_dict(checkpoint['actor_state_dict'])
                        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
                        print(f"체크포인트 불러오기 완료 (에폭 {start_epoch}부터 시작)")
                    else:
                        print("체크포인트가 없습니다.")
            agent.policy.eval()
            # DummyVecEnv에서 실제 환경 가져오기
            real_env = env.envs[0] if hasattr(env, 'envs') else env
            reward, v_list_avg, offset_list_avg, dist_list_avg, avg_reward_list = evaluate(real_env.env, agent, eval_episodes=10, agent_id=AGENT_ID,
                                                        observation_adapter=real_env.observation_adapter,
                                                        max_steps=MAX_NUM_STEPS,
                                                        env_name=config['env_name'],
                                                        human=human,
                                                        name=name,
                                                        seed=seed)
            
            print('\n|Avg Speed:', np.mean(v_list_avg),
                  '\n|Std Speed:', np.std(v_list_avg),
                  '\n|Avg Dist:', np.mean(dist_list_avg),
                  '\n|Std Dist:', np.std(dist_list_avg),
                  '\n|Avg Offset:', np.mean(offset_list_avg),
                  '\n|Std Offset:', np.std(offset_list_avg))

        else:
            save_threshold = train(env, agent, sce, toolModels, start_epoch)
        
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