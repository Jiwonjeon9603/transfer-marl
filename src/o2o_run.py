import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import copy
import json
from learners.multi_task import REGISTRY as le_REGISTRY
from runners.multi_task import REGISTRY as r_REGISTRY
from controllers.multi_task import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.offline_buffer import OfflineBuffer
from components.transforms import OneHot
from itertools import combinations
import torch.nn.functional as F

import numpy as np
import random
import wandb
import uuid

import matplotlib.pyplot as plt
import seaborn as sns
import re


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    results_save_dir = args.results_save_dir

    if args.use_tensorboard and not args.evaluate:
        # only log tensorboard when in training mode
        # though we are always in training mode when we reach here
        tb_exp_direc = os.path.join(results_save_dir, "tb_logs")
        logger.setup_tb(tb_exp_direc)

    # set model save dir
    args.save_dir = os.path.join(results_save_dir, "models", "seed_" + str(args.seed))

    # write config file
    config_str = json.dumps(vars(args), indent=4)
    with open(os.path.join(results_save_dir, "config.json"), "w") as f:
        f.write(config_str)

    # sacred is on by default
    logger.setup_sacred(_run)

    # group_name = _config["task"]
    # wandb_name = group_name

    if "curriculum" in args.task:
        group_name = "Curriculum"
    elif "one" in args.task:
        group_name = "One_task"
    else:
        group_name = "Base"

    wandb_name = _config["task"]

    if "one" in args.task:
        wandb_name = wandb_name + f"_om={args.online_train_tasks}"
    if "curriculum" in args.task:
        wandb_name = group_name + f"_period={args.curriculum_period}"

    _config["job"] = _config["name"]

    wandb.login(relogin=True, key="ad42a1cee565925e2b5065efe7e76c329b954a29")  # jwjeon
    # wandb.login(relogin=True, key="c65dcbd2cd1f30816b9a69b67cf462741ea48880") # mscho
    wandb.init(
        project="MTMA-O2O",
        group = group_name,
        name=wandb_name,
        config=_config,
        id=str(uuid.uuid4()),
    )

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=30)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(main_args, logger, task2runner):
    n_test_runs = max(1, main_args.test_nepisode // main_args.batch_size_run)
    with th.no_grad():
        for task in main_args.test_tasks:
            for _ in range(n_test_runs):
                task2runner[task].run(test_mode=True)

            if main_args.save_replay:
                task2runner[task].save_replay()

            task2runner[task].close_env()

    logger.log_stat("episode", 0, 0)
    logger.print_recent_stats()



def init_tasks(task_list, main_args, logger, buffer_size):
    task2args, task2runner, task2buffer = {}, {}, {}
    task2scheme, task2groups, task2preprocess = {}, {}, {}

    for task in task_list:
        # define task_args
        task_args = copy.deepcopy(main_args)
        task_args.env_args["map_name"] = task
        task2args[task] = task_args

        task_runner = r_REGISTRY[main_args.runner](
            args=task_args, logger=logger, task=task
        )
        task2runner[task] = task_runner

        # Set up schemes and groups here
        env_info = task_runner.get_env_info()
        for k, v in env_info.items():
            setattr(task_args, k, v)

        # Default/Base scheme
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {
                "vshape": (env_info["n_actions"],),
                "group": "agents",
                "dtype": th.int,
            },
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        groups = {"agents": task_args.n_agents}
        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=task_args.n_actions)])
        }

        if task in main_args.online_train_tasks:
            task2buffer[task] = ReplayBuffer(
                scheme,
                groups,
                buffer_size,
                env_info["episode_limit"] + 1,
                preprocess=preprocess,
                device="cpu" if task_args.buffer_cpu_only else task_args.device,
            )
        else:
            task2buffer[task] = ReplayBuffer(
                scheme,
                groups,
                1,
                env_info["episode_limit"] + 1,
                preprocess=preprocess,
                device="cpu" if task_args.buffer_cpu_only else task_args.device,
            )

        # store task information
        task2scheme[task], task2groups[task], task2preprocess[task] = (
            scheme,
            groups,
            preprocess,
        )

    return (
        task2args,
        task2runner,
        task2buffer,
        task2scheme,
        task2groups,
        task2preprocess,
    )


def init_parallel_runner(task_list, main_args, logger, buffer_size):
    task2args, task2runner, task2buffer = {}, {}, {}
    task2scheme, task2groups, task2preprocess = {}, {}, {}

    for task in task_list:
        # define task_args
        task_args = copy.deepcopy(main_args)
        task_args.env_args["map_name"] = task
        task2args[task] = task_args

        task_runner = r_REGISTRY["mt_parallel"](
            args=task_args, logger=logger, task=task
        )
        task2runner[task] = task_runner

        # Set up schemes and groups here
        env_info = task_runner.get_env_info()
        for k, v in env_info.items():
            setattr(task_args, k, v)

        # Default/Base scheme
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {
                "vshape": (env_info["n_actions"],),
                "group": "agents",
                "dtype": th.int,
            },
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        groups = {"agents": task_args.n_agents}
        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=task_args.n_actions)])
        }

        task2buffer[task] = ReplayBuffer(
            scheme,
            groups,
            buffer_size,
            env_info["episode_limit"] + 1,
            preprocess=preprocess,
            device="cpu" if task_args.buffer_cpu_only else task_args.device,
        )

        # store task information
        task2scheme[task], task2groups[task], task2preprocess[task] = (
            scheme,
            groups,
            preprocess,
        )

    return (
        task2args,
        task2runner,
        task2buffer,
        task2scheme,
        task2groups,
        task2preprocess,
    )


def train_sequential(
    train_tasks,
    main_args,
    logger,
    learner,
    task2args,
    task2runner,
    task2offlinedata,
    t_start=0,
):
    ########## start training ##########
    t_env = t_start
    episode = 0  # episode does not matter
    t_max = main_args.offline_tmax
    model_save_time = 0
    last_test_T = 0
    last_log_T = 0
    start_time = time.time()
    last_time = start_time
    test_time_total = 0
    test_start_time = 0

    # get some common information
    batch_size_train = main_args.batch_size
    batch_size_run = main_args.batch_size_run

    # do test before training
    n_test_runs = max(1, main_args.test_nepisode // batch_size_run)
    test_start_time = time.time()
    test_time_total += time.time() - test_start_time
    update_fn = getattr(learner, "update", None)

    while t_env < t_max:
        # shuffle tasks
        np.random.shuffle(train_tasks)
        # train each task
        for task in train_tasks:
            
            episode_sample = task2offlinedata[task].sample(batch_size_train)

            if episode_sample.device != task2args[task].device:
                episode_sample.to(task2args[task].device)
        
            if callable(update_fn):
                terminated = learner.train(
                    episode_sample, t_env / len(train_tasks), episode, task, online=False
                )
            else:
                terminated = learner.train(episode_sample, t_env, episode, task, online=False)

            if terminated is not None and terminated:
                break

            episode += batch_size_run

        t_env += len(train_tasks)
        
        if callable(update_fn):
            update_fn()

        if terminated is not None and terminated:
            logger.console_logger.info(
                f"Terminate training by the learner at t_env = {t_env}. Finish training."
            )
            break

        # Execute test runs once in a while & final evaluation
        if (t_env - last_test_T) / main_args.test_interval >= 1 or t_env >= t_max:
            test_start_time = time.time()

            with th.no_grad():
                for task in main_args.test_tasks:
                    task2runner[task].t_env = t_env
                    for _ in range(n_test_runs):
                        task2runner[task].run(test_mode=True)

            test_time_total += time.time() - test_start_time

            logger.console_logger.info("Step: {} / {}".format(t_env, t_max))
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}. Test time cost: {}".format(
                    time_left(last_time, last_test_T, t_env, t_max),
                    time_str(time.time() - start_time),
                    time_str(test_time_total),
                )
            )
            last_time = time.time()
            last_test_T = t_env

        if main_args.save_model and (
            t_env - model_save_time >= main_args.save_model_interval
            or model_save_time == 0
        ):
            save_path = os.path.join(main_args.save_dir, f"offline_{str(t_env)}")
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))
            learner.save_models(save_path)
            model_save_time = t_env

        if (t_env - last_log_T) >= main_args.log_interval:
            last_log_T = t_env
            logger.log_stat("episode", episode, t_env)
            logger.print_recent_stats()

            wandb.log(
                {
                    "time step": t_env,
                    **{
                        f"{k}": v[-1][1]
                        for k, v in logger.stats.items()
                    },
                }
            )


#### 내가 추가한 부분 #### offline -> online으로 넘어가는 training
def train_online(
    main_args,
    logger,
    learner,
    args,
    episode_runner,
    replaybuffer,
    offlinedata,
    t_start=0,
    use_pcgrad=False,  # False가 기존
):
    ########## start training ##########
    t_env = t_start
    episode = 0  # episode does not matter
    t_max = main_args.online_tmax +  main_args.offline_tmax # main_args.online_tmax
    model_save_time = main_args.offline_tmax # 0
    last_test_T = main_args.offline_tmax # 0
    last_log_T = main_args.offline_tmax #0
    last_curriculum_T = main_args.offline_tmax
    # t_max = main_args.online_tmax
    # model_save_time = 0
    # last_test_T =  0
    # last_log_T = 0
    start_time = time.time()
    last_time = start_time
    test_time_total = 0
    test_start_time = 0

    # get some common information
    batch_size_train = main_args.batch_size
    batch_size_run = main_args.batch_size_run
    if "one" in main_args.task:
        online_tasks = [main_args.online_train_tasks]
    elif "curriculum" in main_args.task:
        online_tasks = []
    else:
        online_tasks = list(main_args.online_train_tasks)

    # do test before training
    n_test_runs = max(1, main_args.test_nepisode // batch_size_run)
    test_start_time = time.time()
    test_time_total += time.time() - test_start_time
    update_fn = getattr(learner, "update", None)
    
    # Import inside function to avoid circular imports or issues if not passed
    from utils.pcgrad import apply_pc_grad

    terminated = None

    # Curriculum state for Easy-to-Hard
    curr_state = {'initialized': False, 'task_sets': [], 'current_set_idx': 0, 'set_start_t': t_start}

    while t_env < t_max:
        # shuffle tasks
        np.random.shuffle(online_tasks)
        if "curriculum" in main_args.task:
            if "Easy-to-Hard" in main_args.task:
                # --- Easy-to-Hard Logic ---
                num_sets = 4
                time_budget_per_set = int((t_max - t_start) / num_sets)
                curriculum_threshold = 0.85  # Win rate threshold to advance early

                # Initialize sets at the start or if not yet initialized
                if not curr_state['initialized']:
                    if main_args.predefined_online_tasks:
                        all_sorted_tasks = main_args.predefined_online_train_tasks
                    
                    else:
                        with th.no_grad():
                            for task in main_args.test_tasks:
                                episode_runner[task].t_env = t_env
                                for _ in range(n_test_runs):
                                    episode_runner[task].run(test_mode=True, heatmap=True)

                        won_mean_list = []
                        for k, v in logger.stats.items():
                            if "test_battle_won_mean" in k:
                                task_name = k.split("/")[0]
                                win_rate = v[-1][1]
                                won_mean_list.append((task_name, win_rate))
                        
                        # Sort Descending (Best -> Worst) for Easy-to-Hard
                        sorted_by_performance = sorted(won_mean_list, key=lambda x: x[1], reverse=True)


                        # Create sets (assuming 12 tasks total, 3 per set)
                        all_sorted_tasks = [t[0] for t in sorted_by_performance]

                    curr_state['task_sets'] = [all_sorted_tasks[i:i+3] for i in range(0, len(all_sorted_tasks), 3)]
                    
                    # Log sets
                    logger.console_logger.info(f"Curriculum Sets Created: {curr_state['task_sets']}")
                    
                    # Initialize state
                    curr_state['current_set_idx'] = 0
                    curr_state['set_start_t'] = t_env
                    online_tasks = curr_state['task_sets'][0]
                    curr_state['initialized'] = True
                    logger.console_logger.info(f"Starting Easy-to-Hard Curriculum with set {curr_state['current_set_idx']}: {online_tasks}")
                
                # Curriculum advancement logic
                if curr_state['initialized'] and curr_state['current_set_idx'] < num_sets - 1:
                    advance_curriculum = False
                    
                    # Time-based advancement
                    if t_env - curr_state['set_start_t'] >= time_budget_per_set:
                        logger.console_logger.info(f"Advancing curriculum due to time budget. Current set {curr_state['current_set_idx']} finished.")
                        advance_curriculum = True
                    
                    # Performance-based advancement (check win rate of current tasks)
                    current_set_win_rates = []
                    for task in online_tasks:
                        for k, v in logger.stats.items():
                            if f"{task}/test_battle_won_mean" == k:
                                current_set_win_rates.append(v[-1][1])
                                break
                    
                    if current_set_win_rates and all(wr >= curriculum_threshold for wr in current_set_win_rates):
                        logger.console_logger.info(f"Advancing curriculum due to high performance ({np.mean(current_set_win_rates):.2f} >= {curriculum_threshold}). Current set {curr_state['current_set_idx']} finished.")
                        advance_curriculum = True

                    if advance_curriculum:
                        curr_state['current_set_idx'] += 1
                        curr_state['set_start_t'] = t_env
                        online_tasks = curr_state['task_sets'][curr_state['current_set_idx']]
                        logger.console_logger.info(f"Advanced to Easy-to-Hard Curriculum set {curr_state['current_set_idx']}: {online_tasks}")
            else:
                # --- Original Curriculum Logic ---
                if t_env == t_start or t_env - last_curriculum_T >= main_args.curriculum_period:
                    if t_env == t_start:
                        with th.no_grad():
                            task_heatmap={}
                            for task in main_args.test_tasks:
                                episode_runner[task].t_env = t_env
                                test_heatmaps = []
                                for _ in range(n_test_runs):  #n_test_runs
                                    episode_runner[task].run(test_mode=True, heatmap=True)
                                    test_heatmaps.append(episode_runner[task].avg_heatmap)
                                task_heatmap[task] = th.mean(th.stack(test_heatmaps, dim=0),dim=0)
                    
                    won_mean_list = []
                    for k, v in logger.stats.items():
                        if "test_battle_won_mean" in k:
                            task_name = k.split("/")[0]
                            win_rate = v[-1][1]
                            won_mean_list.append((task_name, win_rate))
                    random.shuffle(won_mean_list)
                    sorted_by_performance=sorted(won_mean_list, key=lambda x: x[1])
                    
                    # draw_heatmap(task_heatmap, sorted_by_performance, t_env)

                    online_tasks = [t[0] for t in sorted_by_performance[:3]]
                    last_curriculum_T = t_env
            
        collected_tasks_info = []
        
        for task in online_tasks:
            runner = episode_runner[task]
            online_buffer = replaybuffer[task]

            runner.t_env = t_env
            
            episode_batch = runner.run(test_mode=False)
            online_buffer.insert_episode_batch(episode_batch)

            if online_buffer.can_sample(batch_size_train):
                episode_sample = online_buffer.sample(batch_size_train)

                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args[task].device:
                    episode_sample.to(args[task].device)
                
                # Determine if we should update immediately
                should_update = not use_pcgrad

                if callable(update_fn):
                    # ODIS-like learners might use update logic separate from train
                    # But if we use PCGrad, we assume standard learner.train does backward
                    # We pass update flag to learner.train
                    # Note: Not all learners might support 'update' kwarg, we added it to UPDeTLearnerBC
                    try:
                        terminated = learner.train(
                            episode_sample, t_env / len(online_tasks), episode, task, online=True, update=should_update
                        )
                    except TypeError:
                         # Fallback for learners without update arg
                         terminated = learner.train(
                            episode_sample, t_env / len(online_tasks), episode, task, online=True
                        )
                else:
                    try:
                        terminated = learner.train(episode_sample, t_env, episode, task, online=True, update=should_update)
                    except TypeError:
                        terminated = learner.train(episode_sample, t_env, episode, task, online=True)

                # Collect gradients if PCGrad is enabled
                if use_pcgrad:
                    # Collect gradients from learner.params
                    task_grads = []
                    has_grad = False
                    for p in learner.params:
                        if p.grad is not None:
                            task_grads.append(p.grad.clone())
                            has_grad = True
                        else:
                            # We must keep structure consistent
                            task_grads.append(None)
                    
                    if has_grad:
                        collected_tasks_info.append((task, task_grads))
                    
                    # Zero grad for next task
                    learner.optimiser.zero_grad()

                if terminated is not None and terminated:
                    break
            
            episode += batch_size_run

        # Apply PCGrad update if any grads collected
        if use_pcgrad and collected_tasks_info:
            task_names = [x[0] for x in collected_tasks_info]
            task_grads_list = [x[1] for x in collected_tasks_info]
            
            # apply_pc_grad returns the summed projected gradients ready for update
            # It also handles logging internally if logger is provided
            final_grads = apply_pc_grad(task_grads_list, task_names=task_names, logger=logger, t_env=t_env)
            
            # Apply to params
            idx = 0
            for p in learner.params:
                # We should match structure. learner.params is flat list usually.
                if idx < len(final_grads) and final_grads[idx] is not None:
                     if p.requires_grad:
                         p.grad = final_grads[idx]
                idx += 1
            
            learner.optimiser.step()
            learner.optimiser.zero_grad()

        t_env += len(online_tasks)

        
        if callable(update_fn):
            update_fn()

        if terminated is not None and terminated:
            logger.console_logger.info(
                f"Terminate training by the learner at t_env = {t_env}. Finish training."
            )
            break
                                
        # Execute test runs once in a while & final evaluation
        if (t_env - last_test_T) / main_args.test_interval >= 1 or t_env >= t_max:
            test_start_time = time.time()

            # task_heatmap={}
            with th.no_grad():
                for task in main_args.test_tasks:
                    episode_runner[task].t_env = t_env
                    # test_heatmaps = []
                    for _ in range(n_test_runs):
                        episode_runner[task].run(test_mode=True)
            #             test_heatmaps.append(episode_runner[task].avg_heatmap)
            #         task_heatmap[task] = th.mean(th.stack(test_heatmaps, dim=0),dim=0)

            # won_mean_list = []
            # for k, v in logger.stats.items():
            #     if "test_battle_won_mean" in k:
            #         task_name = k.split("/")[0]
            #         win_rate = v[-1][1]
            #         won_mean_list.append((task_name, win_rate))
            # random.shuffle(won_mean_list)
            # sorted_by_performance=sorted(won_mean_list, key=lambda x: x[1])
            
            # draw_heatmap(task_heatmap, sorted_by_performance, t_env)

            test_time_total += time.time() - test_start_time

            logger.console_logger.info("Step: {} / {}".format(t_env, t_max))
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}. Test time cost: {}".format(
                    time_left(last_time, last_test_T, t_env, t_max),
                    time_str(time.time() - start_time),
                    time_str(test_time_total),
                )
            )
            last_time = time.time()
            last_test_T = t_env

        if main_args.save_model and (
            t_env - model_save_time >= main_args.save_model_interval
            or model_save_time == 0
        ):
            save_path = os.path.join(main_args.save_dir, f"online_{str(t_env)}")
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))
            learner.save_models(save_path)
            model_save_time = t_env

        if (t_env - last_log_T) >= main_args.log_interval:
            last_log_T = t_env
            logger.log_stat("episode", episode, t_env)
            logger.print_recent_stats()

            grad_sim_dict = {}
            current_grad_tasks = list(learner.task_grad.keys())
            for task_a, task_b in combinations(current_grad_tasks, 2):
                g1 = learner.task_grad[task_a]
                g2 = learner.task_grad[task_b]
                
                # 코사인 유사도 계산 (벡터가 길기 때문에 F.cosine_similarity 사용)
                sim = F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item()
                
                # 키 이름 예시: "grad_sim/3m_vs_5m_vs_6m"
                grad_sim_dict[f"grad_sim/{task_a}_vs_{task_b}"] = sim
            
            wandb.log(
                {
                    "time step": t_env,
                    **{
                        f"{k}": v[-1][1]
                        for k, v in logger.stats.items()
                    },
                    **{
                        f"selected_task/{task}": (1 if task in online_tasks else 0)
                        for task in main_args.test_tasks
                    },
                    **grad_sim_dict
                }
            )


def draw_heatmap(task_heatmap, sorted_by_performance, t_env):
    save_dir = f"heatmap/tenv_{str(int(t_env))}"
    os.makedirs(save_dir, exist_ok=True)
    for k, v in sorted_by_performance:
        data = task_heatmap[k].cpu().numpy()

        if "_vs_" in k:
            parts = re.findall(r'\d+', k)
            n_allies = int(parts[0])
            n_enemies = int(parts[1])
        else:
            n_allies = int(re.search(r'\d+', k).group())
            n_enemies = n_allies
        # 내 유닛을 제외한 아군 수 (Ally tokens)
        n_ally_tokens = n_allies - 1
        
        # 선을 그을 좌표 계산 (인덱스 기준)
        own_end = 1
        enemy_end = own_end + n_enemies
        ally_end = enemy_end + n_ally_tokens

        plt.figure(figsize=(8, 6))
        
        is_annot = k in ["3m", "4m", "5m", "5m_vs_6m"]
        ax = sns.heatmap(data, annot=is_annot, fmt=".2f", cmap='viridis', vmin=0, vmax=0.4)

        # 3. 구분선 추가 (빨간색 진한 실선)
        # axvline: 세로선 (Key 구분), axhline: 가로선 (Query 구분)
        line_style = {"color": "red", "linewidth": 2.5, "alpha": 0.8}
        
        # Own | Enemy 경계
        ax.axvline(own_end, **line_style)
        ax.axhline(own_end, **line_style)
        
        # Enemy | Ally 경계
        ax.axvline(enemy_end, **line_style)
        ax.axhline(enemy_end, **line_style)
        
        # Ally | Hidden 경계
        ax.axvline(ally_end, **line_style)
        ax.axhline(ally_end, **line_style)

        # 제목 및 저장
        plt.title(f"Task: {k} | Win Rate: {v:.2f}")
        plt.xlabel("Key Tokens (Own-Enemy-Ally-Hidden)")
        plt.ylabel("Query Tokens (Hidden-Ally-Enemy-Ally-Own)")
        
        save_dir = f"heatmap/tenv_{int(t_env)}"
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{save_dir}/{k}_{v:.2f}.png"
        plt.savefig(filename)
        plt.close()

def run_sequential(args, logger):
    # Init runner so we can get env info
    args.n_tasks = len(args.train_tasks)
    # define main_args
    main_args = copy.deepcopy(args)

    all_tasks = list(set(args.train_tasks + args.test_tasks))

    task2args, task2runner, task2buffer, task2scheme, task2groups, task2preprocess = (
        init_tasks(all_tasks, main_args, logger, buffer_size=main_args.buffer_size)
    )
    task2buffer_scheme = {task: task2buffer[task].scheme for task in all_tasks}

    # define mac
    mac = mac_REGISTRY[main_args.mac](
        train_tasks=all_tasks,
        task2scheme=task2buffer_scheme,
        task2args=task2args,
        main_args=main_args,
    )

    for task in main_args.test_tasks:
        task2runner[task].setup(
            scheme=task2scheme[task],
            groups=task2groups[task],
            preprocess=task2preprocess[task],
            mac=mac,
        )

    # define learner
    learner = le_REGISTRY[main_args.learner](mac, logger, main_args)

    if main_args.use_cuda:
        learner.cuda()

    if main_args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(main_args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(
                    main_args.checkpoint_path
                )
            )
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(main_args.checkpoint_path):
            full_name = os.path.join(main_args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if main_args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(
                timesteps, key=lambda x: abs(x - main_args.load_step)
            )

        model_path = os.path.join(main_args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)

        if main_args.evaluate or main_args.save_replay:
            evaluate_sequential(main_args, logger, task2runner)
            return
        
    # initialize training data for each task
    task2offlinedata = {}
    for task in main_args.train_tasks:
        # create offline data buffer
        task2offlinedata[task] = OfflineBuffer(
            task,
            main_args.train_tasks_data_quality[task],
            data_folder=main_args.offline_data_name,
            offline_data_size=args.offline_data_size,
            random_sample=args.offline_data_shuffle,
        )

    logger.console_logger.info(
        "Beginning multi-task offline training with {} timesteps for each task".format(
            main_args.offline_tmax
        )
    )


    if not main_args.learn_only_online:
        train_sequential(
            main_args.train_tasks,
            main_args,
            logger,
            learner,
            task2args,
            task2runner,
            task2offlinedata,
        )

        # save the final model
        if main_args.save_model:
            save_path = os.path.join(main_args.results_save_dir, "Offline", "models", "seed_" + str(main_args.seed), str(main_args.offline_tmax))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving final models to {}".format(save_path))
            learner.save_models(save_path)


    # logger.console_logger.info("Re-initializing runners and buffers for online phase")

    # task2args_online, task2runner_online, task2buffer_online, task2scheme_online, task2groups_online, task2preprocess_online = (
    #     init_tasks(all_tasks, main_args, logger, buffer_size=main_args.buffer_size)
    # )
    
    # for task in all_tasks:
    #     task2runner_online[task].setup(
    #         scheme=task2scheme_online[task],
    #         groups=task2groups_online[task],
    #         preprocess=task2preprocess_online[task],
    #         mac=mac,   # 기존 mac 그대로
    #     )

    logger.console_logger.info(
        f"Beginning multi-task online training with {main_args.online_tmax} timesteps"
    )

    if main_args.use_lora:
        # Apply LoRA
        from utils.lora import inject_lora, freeze_model
        from utils.pcgrad import apply_pc_grad
        logger.console_logger.info("Injecting LoRA ...")
        
        # You can customize r and alpha if they are in args, otherwise default
        lora_r = getattr(main_args, "lora_r", 16)
        lora_alpha = getattr(main_args, "lora_alpha", 32)
        
        # Freeze the entire agent first (common part)
        freeze_model(learner.mac.agent)
        freeze_model(learner.target_mac.agent)
        
        inject_lora(learner.mac.agent, r=lora_r, alpha=lora_alpha)
        inject_lora(learner.target_mac.agent, r=lora_r, alpha=lora_alpha)
        
        # Move to CUDA if needed
        if main_args.use_cuda:
            learner.mac.cuda()
            learner.target_mac.cuda()
            
        
        # Re-initialize optimizer with new parameters (filtering for requires_grad=True)
        learner.params = list(filter(lambda p: p.requires_grad, learner.mac.parameters()))
        
        if main_args.optim_type.lower() == "rmsprop":
            learner.optimiser = th.optim.RMSprop(params=learner.params, lr=main_args.lr, alpha=main_args.optim_alpha, eps=main_args.optim_eps, weight_decay=main_args.weight_decay)
        elif main_args.optim_type.lower() == "adam":
            learner.optimiser = th.optim.Adam(params=learner.params, lr=main_args.lr, weight_decay=main_args.weight_decay)
        
        logger.console_logger.info(f"LoRA injected with r={lora_r}, alpha={lora_alpha}. Optimizer re-initialized.")
        

    # for task in args.test_tasks:
    #     task2runner[task].close_env()

    train_online(
        main_args,
        logger,
        learner,
        task2args,
        task2runner,
        # parallel_runner,
        task2buffer,   # ★ online replay
        task2offlinedata,
        t_start= main_args.offline_tmax,
        use_pcgrad=main_args.use_pcgrad,
    )

    wandb.finish()

    # save the final model
    if main_args.save_model:
        save_path = os.path.join(main_args.results_save_dir, "Online", "models", "seed_" + str(main_args.seed), str(main_args.online_tmax))
        os.makedirs(save_path, exist_ok=True)
        logger.console_logger.info("Saving final models to {}".format(save_path))
        learner.save_models(save_path)

    for task in args.test_tasks:
        task2runner[task].close_env()
    logger.console_logger.info(f"Finished Training")


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config
