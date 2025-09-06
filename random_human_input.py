import random
import time

ACTIONS = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]

while True:
    # 1초에 한 번, 50% 확률로 무작위 액션 입력
    if random.random() < 0.5:
        action = random.choice(ACTIONS)
        with open("human_action.txt", "w") as f:
            f.write(action)
        print(f"무작위 입력: {action}")
    time.sleep(1)