SEEDS=(0)
HIGH_STEPS=(3)
TASKS=(
    # "marine-hard-expert"
    "marine-hard-medium"
    # "marine-hard-medium-expert"
    # "marine-hard-medium-replay"

    # "marine-easy-expert"
    # "marine-easy-medium"
    # "marine-easy-medium-expert"
    # "marine-easy-medium-replay"

    # "stalker-zealot-expert"
    # "stalker-zealot-medium"
    # "stalker-zealot-medium-expert"
    # "stalker-zealot-medium-replay"


)

for t in "${TASKS[@]}"; do
    for i in "${SEEDS[@]}"; do
        CUDA_VISIBLE_DEVICES=0 python src/main.py --baseline_run --config=stairs --env-config=sc2_offline --task-config=$t --seed=$i
    done
done

