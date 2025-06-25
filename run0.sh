CUDA_VISIBLE_DEVICES=0 taskset -c 0-15,32-47 python src/main.py --mto --config=odis --env-config=sc2_offline --task-config=toy0 --seed=0
