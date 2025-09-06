# :tada: _T-ITS ACCEPTED!_ :confetti_ball:

# :page_with_curl: Safe Human-in-the-loop RL (SafeHiL-RL) with Shared Control for End-to-End Autonomous Driving

# :fire: Source Code Released! :fire:

## [[**T-ITS**]](https://ieeexplore.ieee.org/document/10596046) | [[**arXiv**]](https://www.researchgate.net/publication/382212078_Safety-Aware_Human-in-the-Loop_Reinforcement_Learning_With_Shared_Control_for_Autonomous_Driving)

## 🎯 실험 소개 및 연구 목적

### 📋 연구 배경
자율주행 기술의 발전과 함께 인공지능(AI)과 인간의 협력 시스템에 대한 관심이 높아지고 있습니다. 특히, LLM(Large Language Model)과 강화학습 에이전트의 협력 시스템에서 인간의 개입이 시스템 성능에 미치는 영향을 정량적으로 평가하는 것은 매우 중요한 연구 분야입니다.

### 🔬 실험 목표
본 연구는 다음과 같은 핵심 목표를 가지고 진행됩니다:

1. **LLM-에이전트 협력 시스템 성능 평가**
   - LLM과 강화학습 에이전트 간의 협력 의사결정 구조 분석
   - 다양한 협력 모드(LLM+Agent, LLM+Agent+Human) 간의 성능 비교
   - 시스템 안정성과 학습 효율성 측면에서의 정량적 평가

2. **인간 개입 효과 분석**
   - 인간의 개입이 자율주행 시스템 성능에 미치는 영향 정량화
   - 안전성 향상과 학습 효율성 간의 균형점 탐색
   - 인간-AI 공유 제어(Shared Control) 메커니즘의 효과 검증

3. **다양한 의사결정 구조 비교**
   - SaHiL (Safety-Aware Human-in-the-Loop)
   - PHIL (Policy-based Human-in-the-Loop) 
   - HIRL (Human-in-the-Loop Reinforcement Learning)
   - SAC (Soft Actor-Critic) 기반 시스템

### 🛠️ 실험 방법론

#### 실험 환경
- **시뮬레이터**: SMARTS (Scalable Multi-Agent Reinforcement Learning Training School)
- **시나리오**: 고속도로 주행 환경 (straight, straight_with_left_turn 등)
- **평가 지표**: 
  - 안전성: 충돌률, 도로 이탈률
  - 효율성: 평균 보상, 목표 달성률
  - 인간성: 차선 중심 이탈도, 급격한 조향 변화

#### 실험 구성
1. **LLM+Agent 시스템**: LLM과 강화학습 에이전트의 협력 의사결정
2. **LLM+Agent+Human 시스템**: 인간 개입이 추가된 3자 협력 시스템
3. **제어 권한 할당**: 상황에 따른 동적 권한 분배 메커니즘

#### 평가 메트릭
- **성능 지표**: 평균 보상, 성공률, 안전성 점수
- **학습 효율성**: 수렴 속도, 정책 안정성
- **인간 개입 효과**: 개입 빈도, 개입 효과성, 시스템 안정성

### 🎯 기대 효과

#### 이론적 기여
- **인간-AI 협력 시스템의 정량적 평가 프레임워크** 구축
- **안전성 인식 강화학습** 기법의 효과 검증
- **공유 제어 메커니즘**의 최적화 방법론 제시

#### 실용적 기여
- **자율주행 시스템의 안전성 향상**을 위한 구체적 방안 제시
- **인간 개입의 효과적 활용** 방법론 개발
- **실시간 의사결정 시스템**의 신뢰성 증대

### 📊 실험 결과 분석
실험 결과는 `results/` 디렉토리에서 확인할 수 있으며, 다음과 같은 분석을 제공합니다:
- **에포크별 성능 변화**: 400, 600, 700, 775, 820 에포크에서의 성능 비교
- **시스템별 상대적 성능**: LLM+Agent vs LLM+Agent+Human 시스템 비교
- **안전성 지표**: 충돌률, 도로 이탈률, 안전 마진 등

---

:dizzy: As a **_pioneering work considering guidance safety_** within the human-in-the-loop RL paradigm, this work introduces a :fire: **_curriculum guidance mechanism_** :fire: inspired by the pedagogical principle of whole-to-part patterns in human education, aiming to standardize the intervention process of human participants.

:red_car: SafeHil-RL is designed to prevent **_policy oscillations or divergence_** caused by **_inappropriate or degraded human guidance_** during interventions using the **_human-AI shared autonomy_** technique, thereby improving learning efficiency, robustness, and driving safety.

:wrench: Realized in SMARTS simulator with Ubuntu 20.04 and Pytorch. 

Email: wenhui001@e.ntu.edu.sg

# Framework

<p align="center">
<img src="https://github.com/OscarHuangWind/Human-in-the-loop-RL/blob/master/presentation/framework.png" height= "450" width="900">
</p>

# Frenet-based Dynamic Potential Field (FDPF)
<p float="left">
  <img src="https://github.com/OscarHuangWind/Human-in-the-loop-RL/blob/master/presentation/FDPF_scenarios.png" height= "140" />
  <img src="https://github.com/OscarHuangWind/Human-in-the-loop-RL/blob/master/presentation/FDPF_bound.png" height= "140" /> 
  <img src="https://github.com/OscarHuangWind/Human-in-the-loop-RL/blob/master/presentation/FDPF_obstacle.png" height= "140" />
  <img src="https://github.com/OscarHuangWind/Human-in-the-loop-RL/blob/master/presentation/FDPF_final.png" height= "140" />
</p>

# Demonstration (accelerated videos)

## Lane-change Performance
https://github.com/OscarHuangWind/Human-in-the-loop-RL/assets/41904672/690b4b44-ac57-4ce1-890b-57ac125cef63
## Uncooperative Road User
https://github.com/OscarHuangWind/Human-in-the-loop-RL/assets/41904672/52b2ec4b-8cd4-4b9d-a3a9-70bbd3b77157
## Cooperative Road User
https://github.com/OscarHuangWind/Human-in-the-loop-RL/assets/41904672/02f95274-80cc-4e6b-8a5b-edfcbbd4d0a6
## Unobserved Road Structure
https://github.com/OscarHuangWind/Human-in-the-loop-RL/assets/41904672/bb493f9c-d2c9-4db5-b034-ad456ef96c8a

# User Guide

## Clone the repository.
cd to your workspace and clone the repo.
```
git clone https://github.com/OscarHuangWind/Safe-Human-in-the-Loop-RL.git
```

## Create a new Conda environment.
cd to your workspace:
```
conda env create -f environment.yml
```

## Activate virtual environment.
```
conda activate safehil-rl
```

## Install Pytorch
Select the correct version based on your cuda version and device (cpu/gpu):
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Install the SMARTS.
```
# Download SMARTS

git clone https://github.com/huawei-noah/SMARTS.git

cd <path/to/SMARTS>

# Important! Checkout to comp-1 branch
git checkout comp-1

# Install the system requirements.
bash utils/setup/install_deps.sh

# Install smarts.
pip install -e '.[camera_obs,test,train]'

# Install extra dependencies.
pip install -e .[extras]
```

## Build the scenario.
```
cd <path/to/Safe-Human-in-the-loop-RL>
scl scenario build --clean scenario/straight/
```

## Visulazation
```
scl envision start
```
Then go to http://localhost:8081/

## Training
Modify the sys path in **main.py** file, and run:
```
python main.py
```

## Human Guidance
Change the model in **main.py** file to SaHiL/PHIL/HIRL, and run:
```
python main.py
```
Check the code in keyboard.py to get idea of keyboard control.

Alternatively, you can use G29 set to intervene the vehicle control, check the lines from 177 to 191 in main.py file for the details.

The "Egocentric View" is recommended for the human guidance.

## Evaluation
Edit the mode in config.yaml as evaluation and run:
```
python main.py
```




