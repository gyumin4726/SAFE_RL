from openai import OpenAI
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
    name = ''
    description = ''

    if action[2] < -0.1:
        name += ACTIONS_ALL[0]
        description += ACTIONS_DESCRIPTION[0]
    elif action[2] > 0.1:
        name += ACTIONS_ALL[1]
        description += ACTIONS_DESCRIPTION[1]
    else:
        name += ACTIONS_ALL[2]
        description += ACTIONS_DESCRIPTION[2]

    name += ', '
    description += ', '

    if action[0] > 0:
        name += ACTIONS_ALL[3]
        description += ACTIONS_DESCRIPTION[3]
    else:
        name += ACTIONS_ALL[4]
        description += ACTIONS_DESCRIPTION[4]   

    return name, description

def send_to_chatgpt(last_action, current_scenario, sce):
    client = OpenAI(api_key=API_KEY)
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
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return completion.choices[0].message
