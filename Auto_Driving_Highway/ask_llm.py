import google.generativeai as genai
import pre_prompt
from logging import lastResort
import os

from customTools import ACTIONS_ALL, ACTIONS_DESCRIPTION

current_file_abs_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_abs_path)
key_path = os.path.join(current_dir, 'MY_KEY.txt')

with open(key_path, 'r', encoding='utf-8-sig') as f:
    api_key = f.read().strip()
API_KEY = api_key

def get_action_info(action):
    throttle, brake, steering = action
    if steering <= -0.1:
        return "LANE_LEFT", "change direction to the left"
    elif steering >= 0.1:
        return "LANE_RIGHT", "change direction to the right"
    elif throttle > 0.2:
        return "FASTER", "accelerate the vehicle"
    elif brake > 0.1:
        return "SLOWER", "decelerate the vehicle"
    elif throttle > 0.1:
        return "IDLE", "remain in the current direction"
    else:
        return "IDLE", "remain in the current direction"

# 디버깅용 함수: normalize_action

def normalize_action(val):
    if isinstance(val, (set, list, tuple)):
        val = list(val)[0] if val else ""
    if isinstance(val, dict):
        val = list(val.keys())[0] if val else ""
    return str(val).strip().upper()

def send_to_chatgpt(last_action, current_scenario, sce):
    genai.configure(api_key=API_KEY)
    message_prefix = pre_prompt.SYSTEM_MESSAGE_PREFIX
    traffic_rules = pre_prompt.get_traffic_rules()
    decision_cautions = pre_prompt.get_decision_cautions()
    action_name, action_description = get_action_info(last_action)

    prompt = (f"{message_prefix}"
              f"You, the 'ego' car, are now driving on a highway. You have already driven for {sce.frame} seconds.\n"
              "There are several rules you need to follow when you drive on a highway:\n"
              f"{traffic_rules}\n\n"
              "Here are your attention points:\n"
              f"{decision_cautions}\n\n"
              "Once you make a final decision, output it in the following format:\n"
              "```\n"
              "Final Answer: \n"
              "    \"decision\": {\"<ego car's decision, ONE of the available actions>\"},\n"
              "```\n")
    user_prompt = (f"The decision made by the agent LAST time step was `{action_name}` ({action_description}).\n\n"
                   "Here is the current scenario:\n"
                   f"{current_scenario}\n\n")
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    response = model.generate_content([
        {"role": "user", "parts": [prompt + user_prompt]}
    ])
    return response.text
