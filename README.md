# Offline-to-Online (O2O) Multi-Task Training

`src/o2o_run.py` 기반 학습 가이드. 하나의 실행에서 **offline 학습 → online 학습**이 이어서 진행되며,
online 단계에서 어떤 task를 학습할지를 task config로 제어한다.

---

## 1. 전체 흐름

```
offline 학습 (offline_tmax step)        online 학습 (online_tmax step)
┌──────────────────────────────┐   ┌──────────────────────────────────────┐
│ offline 데이터셋에서 BC 학습   │ → │ 선택된 task로 환경과 상호작용하며 학습 │
│ t = 0 ... offline_tmax        │   │ t = offline_tmax ... offline+online   │
└──────────────────────────────┘   └──────────────────────────────────────┘
```

- 기본 설정: **offline 30,000 step → online 30,000 step** (총 t = 0 ~ 60,000)
- offline 단계는 `learn_only_online: True`로 건너뛸 수 있다 (→ §5, checkpoint 사용).
- online 단계의 task 선택 방식은 **task config 이름**으로 결정된다 (→ §3, §4).

관련 설정은 [src/config/algs/updet-o2o.yaml](src/config/algs/updet-o2o.yaml)에 있다.

---

## 2. 실행 명령 템플릿

```bash
python src/main.py --o2o_run \
  --config=updet-o2o \
  --env-config=sc2_offline \
  --task-config=<TASK_CONFIG> \
  --seed=1
```

| 인자 | 의미 |
|---|---|
| `--o2o_run` | `o2o_run.py`를 실행 (run_file 지정, `=` 없는 첫 `--` 인자) |
| `--config=updet-o2o` | 알고리즘 config ([algs/updet-o2o.yaml](src/config/algs/updet-o2o.yaml)) |
| `--env-config=sc2_offline` | 환경 config. `sc2_offline`은 `test_interval=500` |
| `--task-config=<...>` | task config ([config/tasks/](src/config/tasks/)) — online task 선택을 결정 |
| `--seed=N` | 랜덤 시드 |

> 임의의 config 값은 CLI로 덮어쓸 수 있다. 예: `--online_tmax=10000`, `--learn_only_online=True`,
> `--save_model=True`. 값은 파이썬 리터럴로 평가된다.

---

## 3. 특정 task로 online 학습하기 (predefined)

curriculum이 아닌 경우, online에서 학습할 task는 **고정 목록**이다.
모든 비-curriculum 경우는 하나의 task config로 통일되어 있다:
[config/tasks/marine-hard-medium-o2o.yaml](src/config/tasks/marine-hard-medium-o2o.yaml).

`online_train_tasks` 리스트만 바꾸면 학습할 task가 1개든, 3개든, 4개든, 전체든 상관없이 동작한다.

```yaml
# marine-hard-medium-o2o.yaml
online_train_tasks: ["7m_vs_8m"]                              # 1개
# online_train_tasks: ["3m", "5m_vs_6m", "9m_vs_10m"]        # 3개 (same)
# online_train_tasks: ["8m_vs_9m", "10m_vs_11m", "13m_vs_15m"] # 3개 (hard)
# online_train_tasks: ["3m", "4m", ... , "13m_vs_15m"]       # 전체 (base)
```

실행:

```bash
python src/main.py --o2o_run --config=updet-o2o --env-config=sc2_offline \
  --task-config=marine-hard-medium-o2o --seed=1
```

CLI에서 직접 덮어쓰는 것도 가능하다 (yaml 수정 없이):

```bash
python src/main.py --o2o_run --config=updet-o2o --env-config=sc2_offline \
  --task-config=marine-hard-medium-o2o \
  --online_train_tasks="['7m_vs_8m','8m_vs_9m']" --seed=1
```

> **wandb run 이름**에 선택한 task 목록이 포함되어(`..._tasks=[...]`) 선택별로 run이 구분된다.

---

## 4. Curriculum으로 online 학습하기

task 이름에 `curriculum`이 들어가면 online task가 **자동 선택**된다. 두 방식이 있다.

### (a) Original — 매 주기마다 "가장 못하는 3개" 동적 선택

[config/tasks/marine-hard-medium-o2o-curriculum.yaml](src/config/tasks/marine-hard-medium-o2o-curriculum.yaml)

- `curriculum_period`(기본 1,000) step마다 전체 test task의 win-rate를 정렬해
  **성능이 가장 낮은 3개**를 online task로 선택한다.
- win-rate는 `test_interval`마다 갱신된 값을 사용한다. `sc2_offline`은 `test_interval=500`이라
  `curriculum_period=1000`과 잘 맞물린다.

```bash
python src/main.py --o2o_run --config=updet-o2o --env-config=sc2_offline \
  --task-config=marine-hard-medium-o2o-curriculum --seed=1
```

### (b) Easy-to-Hard / Hard-to-Easy — 난이도 순서대로 set 진행

[easy-to-hard](src/config/tasks/marine-hard-medium-o2o-curriculum-easy-to-hard.yaml) ·
[hard-to-easy](src/config/tasks/marine-hard-medium-o2o-curriculum-hard-to-easy.yaml) ·
[ETH-seed0~4](src/config/tasks/)

- online 시작 시 전체 task를 win-rate로 정렬하고 **3개씩 4개 set**으로 묶는다.
  - `Easy-to-Hard` / `ETH`: 잘하는 것부터 (내림차순)
  - `Hard-to-Easy`: 못하는 것부터 (오름차순)
- 각 set은 `(online_tmax)/4` step 경과 시, 또는 set 내 모든 task의 win-rate ≥ 0.85일 때 다음 set으로 진행.
- `predefined_online_tasks: True`로 두면 평가 대신 yaml의 `predefined_online_train_tasks` 순서를 그대로 사용한다.

```bash
python src/main.py --o2o_run --config=updet-o2o --env-config=sc2_offline \
  --task-config=marine-hard-medium-o2o-curriculum-easy-to-hard --seed=1
```

> 분기는 task 이름으로 결정된다 ([o2o_run.py](src/o2o_run.py)): 이름에 `Easy-to-Hard`/`Hard-to-Easy`/`ETH`가
> 있으면 (b), 그 외 `curriculum`이면 (a) Original.

---

## 5. Offline checkpoint 재사용하기 (offline 건너뛰기)

이미 offline 학습된 모델이 있으면 offline 단계를 건너뛰고 online만 돌릴 수 있다.

### 5.1 checkpoint 생성 (offline 모델 저장)

offline부터 학습하면서 모델을 저장하려면 `save_model=True`가 필요하다 (기본값 False).

```bash
python src/main.py --o2o_run --config=updet-o2o --env-config=sc2_offline \
  --task-config=marine-hard-medium-o2o --learn_only_online=False \
  --save_model=True --seed=1
```

offline 최종 모델은 다음 경로에 저장된다:

```
results/<task>/<name>/dropout_<token_dropout>/Offline/models/seed_<seed>/<offline_tmax>/
# 예: results/marine-hard-medium-o2o/updet-bc/dropout_0.1/Offline/models/seed_1/30000/
```

### 5.2 checkpoint로 online만 학습

`learn_only_online: True`로 offline을 건너뛰고, `checkpoint_path`로 위 `seed_<seed>` 폴더를 가리킨다
(폴더 안의 숫자 디렉토리 중 `load_step`에 가장 가까운, 기본은 최대 step을 로드).

```bash
python src/main.py --o2o_run --config=updet-o2o --env-config=sc2_offline \
  --task-config=marine-hard-medium-o2o \
  --learn_only_online=True \
  --checkpoint_path=results/marine-hard-medium-o2o/updet-bc/dropout_0.1/Offline/models/seed_1 \
  --seed=1
```

> checkpoint_path의 마지막 경로는 **숫자 이름의 하위 폴더**(`30000` 등)를 포함해야 로드된다.
> 평가만 하고 종료하려면 `--evaluate=True`를 추가한다.

---

## 6. 주요 설정값 ([algs/updet-o2o.yaml](src/config/algs/updet-o2o.yaml))

| 키 | 기본값 | 설명 |
|---|---|---|
| `offline_tmax` | 30000 | offline 학습 step |
| `online_tmax` | 30000 | online 학습 step |
| `learn_only_online` | False | True면 offline 건너뜀 (checkpoint와 함께 사용) |
| `predefined_online_tasks` | False | curriculum(b)에서 사전 정의 순서 사용 여부 |
| `curriculum_period` | 1000 | Original curriculum의 재선택 주기 |
| `use_pcgrad` | False | online에서 task 간 PCGrad gradient projection |
| `use_lora` | False | online 단계에 LoRA 주입 |

환경 config([sc2_offline.yaml](src/config/envs/sc2_offline.yaml))의 `test_interval`(=500),
`log_interval`(=500), `save_model_interval`(=30000)도 함께 적용된다.

---

## 7. 출력

- **모델**: offline `.../Offline/models/seed_<seed>/<offline_tmax>/`, online `.../Online/models/seed_<seed>/<online_tmax>/`
  (`save_model=True`일 때). 중간 저장은 `.../models/seed_<seed>/offline_<t>/`, `online_<t>/`.
- **wandb**: project `MTMA-O2O`, group은 curriculum이면 `Curriculum` 아니면 `Base`.
  run 이름에 task와 (비-curriculum의 경우) `online_train_tasks` 목록, `time-step`이 포함된다.
- **로그**: `results/<task>/<name>/dropout_<token_dropout>/` 아래 sacred / tb_logs.
