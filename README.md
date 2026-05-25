# SafeHiL-LLM

LLM과 DRL이 협력하는 자율주행 시스템에서 **불완전한 인간 개입(Imperfect Human-in-the-Loop)** 이 성능과 안전성에 미치는 영향을 분석한 실험 프로젝트입니다.

기존 HITL·RLHF 연구가 암묵적으로 가정하는 *"인간 입력 = Ground Truth"* 가설을, 잘못된 인간 개입 시나리오에서 정량적으로 재검증합니다.

> **Project 3 of R&Dix PhysAI Team Project (2025)**
> Base framework: [SafeHiL-RL (Huang et al., T-ITS 2024)](https://ieeexplore.ieee.org/document/10596046) · Simulator: [SMARTS](https://github.com/huawei-noah/SMARTS)

## Overview

본 연구는 두 가지 의사결정 구조를 직접 비교합니다.

| | **System A** | **System B** |
| --- | --- | --- |
| 구성 | LLM + RL Agent | LLM + RL Agent + Imperfect Human |
| 동작 | LLM이 안전 행동을 제안하고 RL이 제어 | System A에 잘못된/무작위 인간 개입을 주입 |

LLM은 단순 보조가 아니라 (1) 도로 맥락·차량 의도 같은 **고수준 상황 추론**, (2) 행동의 **안전성 설명(XAI)**, (3) 사전 지식 기반 **탐색 공간 축소** 역할을 담당합니다.

## Results

SMARTS highway 시나리오, Epoch 820 기준 10회 평가 결과입니다.

| System | Avg Reward | Success | Collision | Off-Road |
| --- | ---: | ---: | ---: | ---: |
| LLM + RL Agent | **+0.51** | **8 / 10** | 2 | 0 |
| LLM + RL + Imperfect Human | −4.90 | 0 / 10 | 3 | 7 |

- LLM + RL 협력만으로도 안정적인 정책 학습이 가능했습니다.
- 불완전한 인간 개입은 단순 노이즈가 아니라 **충돌·이탈을 유발하는 안전성 저해 요인**으로 작용했습니다.
- 인간을 *Ground Truth* 가 아닌 *잠재적 노이즈 소스* 로 가정하고 LLM이 필터링·권한 조정을 수행하는 검증 기반 HIL 설계가 필요함을 시사합니다.

## Installation

```bash
git clone https://github.com/gyumin4726/SAFE_RL.git
cd SAFE_RL

conda env create -f environment.yml
conda activate safehil-rl
```

SMARTS 시뮬레이터 설치:

```bash
git clone https://github.com/huawei-noah/SMARTS.git
cd SMARTS && git checkout comp-1
bash utils/setup/install_deps.sh
pip install -e '.[camera_obs,test,train]'
pip install -e '.[extras]'
```

## Usage

시나리오 빌드:

```bash
scl scenario build --clean scenario/straight/
```

시각화 (선택):

```bash
scl envision start   # http://localhost:8081/
```

학습 / 평가:

```bash
python train_agent.py
```

실행 모드(학습 / 평가)와 사용 모델(SAC / HIRL / PHIL / SaHiL)은 `config.yaml` 에서 조정합니다.

## Project Structure

```
.
├── train_agent.py            # 메인 학습·평가 엔트리포인트
├── main.py                   # SafeHiL-RL 원본 학습 루프
├── drl_agent.py              # SAC 기반 RL 에이전트
├── Network.py                # 정책·가치 네트워크
├── authority_allocation.py   # LLM·RL·Human 권한 분배
├── dynamic_potential_field.py# FDPF 안전 필드
├── random_human_input.py     # Imperfect Human 시뮬레이션 (Experiment 4)
├── keyboard.py               # 키보드 기반 인간 개입
├── scenario/                 # SMARTS 시나리오 정의
└── config.yaml               # 실행 설정
```

## Stack

PyTorch · Python 3.9 · SMARTS · SAC

## Acknowledgments

본 프로젝트는 [Safe Human-in-the-Loop RL (Huang et al.)](https://github.com/OscarHuangWind/Safe-Human-in-the-Loop-RL) 의 SAC + FDPF 권한 할당 구조를 기반으로, LLM Safety Explainer 통합 및 Imperfect Human 시나리오를 확장 구현하였습니다.
