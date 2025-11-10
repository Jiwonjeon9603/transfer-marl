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
import uuid

from learners.multi_task import REGISTRY as le_REGISTRY
from runners.multi_task import REGISTRY as r_REGISTRY
from controllers.multi_task import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.offline_buffer import OfflineBuffer
from components.transforms import OneHot

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import wandb

from sklearn.manifold import TSNE
from collections import defaultdict
from sklearn.cluster import DBSCAN

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
    if getattr(args, "s0_filter", False):
        wandb_name = f"agent={args.algo_name}-s0_thresh={str(args.s0_filter_threshold)}-s0_topk={str(args.s0_filter_topk)}"
    else:
        wandb_name = f"agent={args.algo_name}"
    _config["job"] = _config["name"]
    # _config = {k: str(v) for k, v in _config.items()}
    wandb.login(relogin=True, key="ad42a1cee565925e2b5065efe7e76c329b954a29")  # jwjeon
    # wandb.login(relogin=True, key="c65dcbd2cd1f30816b9a69b67cf462741ea48880") # mscho
    wandb.init(
        project="test-Proposal-MTMA-21",
        group=_config["task"],
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


def draw_attention_heatmap(
    attention, task, num_steps_to_plot, batch_idx, first_dead, main_args, layer
):
    if task == "3m":
        n_ally = 2
        n_enemy = 3
        vmax = 0.4
    elif task == "4m":
        n_ally = 3
        n_enemy = 4
        vmax = 0.4
    elif task == "5m_vs_6m":
        n_ally = 4
        n_enemy = 6
        vmax = 0.3
    elif task == "9m_vs_10m":
        n_ally = 8
        n_enemy = 10
        vmax = 0.1
    else:
        n_ally = 1
        n_enemy = 1

    maps = attention.cpu().numpy()  # [T, A, H, W]
    T, A, H, W = maps.shape

    time_indices = np.linspace(0, T - 1, num_steps_to_plot, dtype=int)

    fig, axes = plt.subplots(
        A, num_steps_to_plot, figsize=(num_steps_to_plot * 2.5, A * 2)
    )

    if A == 1:
        axes = np.expand_dims(axes, 0)
    if num_steps_to_plot == 1:
        axes = np.expand_dims(axes, 1)

    boundaries = [0, 1, 1 + n_enemy, 1 + n_enemy + n_ally, H]
    labels = ["own", "enemy", "ally", "hidden"]

    # 🎯 colorbar는 alpha=1.0 기준으로 생성
    example_heatmap = maps[0, 0]
    fig_for_cbar = plt.figure()
    ax_for_cbar = fig_for_cbar.add_subplot(111)
    im_cbar = ax_for_cbar.imshow(
        example_heatmap,
        cmap="viridis",
        interpolation="nearest",
        aspect="auto",
        alpha=1.0,
    )
    plt.close(fig_for_cbar)

    vmin = 0

    for agent in range(A):
        for i, t in enumerate(time_indices):
            ax = axes[agent][i]
            heatmap = maps[t, agent]
            # ✅ first_dead 이후면 연하게
            alpha = 1.0 if t < first_dead[agent] else 0.3

            im = ax.imshow(
                heatmap,
                vmin=vmin,
                vmax=vmax,
                cmap="viridis",
                interpolation="nearest",
                aspect="auto",
                alpha=alpha,
            )

            # 기본 토큰 간 grid
            ax.set_xticks(np.arange(W + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(H + 1) - 0.5, minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
            ax.tick_params(which="minor", bottom=False, left=False)
            ax.set_xticks([])
            ax.set_yticks([])

            # 빨간 경계선
            for b in boundaries[1:-1]:
                ax.axvline(b - 0.5, color="red", linewidth=1.5)
                ax.axhline(b - 0.5, color="red", linewidth=1.5)

            # 라벨
            for idx in range(len(labels)):
                start = boundaries[idx]
                end = boundaries[idx + 1]
                center = (start + end - 1) / 2
                ax.text(
                    center,
                    H + 0.2,
                    labels[idx],
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="black",
                    transform=ax.transData,
                )

            if agent == 0:
                ax.set_title(f"t={t}")
            if i == 0:
                ax.set_ylabel(f"Agent {agent}", rotation=90, fontsize=10)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im_cbar, cax=cbar_ax, label="Attention weight")

    plt.suptitle(f"Attention (Batch {batch_idx})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    # 저장
    if getattr(main_args, "high_hidden_dropout", False):
        fig_name = (
            "Attention_Heatmap/"
            + main_args.name
            + "_"
            + str(main_args.token_dropout)
            + "_HD"
            + "/"
            + str(layer)
            + "st-layer"
        )
    else:
        fig_name = (
            "Attention_Heatmap/"
            + main_args.name
            + "_"
            + str(main_args.token_dropout)
            + "/"
            + str(layer)
            + "st-layer"
        )
    save_dir = os.path.join(os.getcwd(), fig_name)
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{task}_batch_{batch_idx}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def draw_mean_attention_heatmap(
    attention, task, num_steps_to_plot, batch_idx, first_dead, main_args, layer
):
    import os
    import torch as th
    import numpy as np
    import matplotlib.pyplot as plt

    # task 설정에 따른 토큰 수
    if task == "3m":
        n_ally = 2
        n_enemy = 3
        vmax = 0.4
    elif task == "4m":
        n_ally = 3
        n_enemy = 4
        vmax = 0.4
    elif task == "5m_vs_6m":
        n_ally = 4
        n_enemy = 6
        vmax = 0.3
    elif task == "9m_vs_10m":
        n_ally = 8
        n_enemy = 10
        vmax = 0.1
    else:
        n_ally = 1
        n_enemy = 1

    maps = attention.cpu().numpy()  # [T, A, H, W]
    T, A, H, W = maps.shape

    time_indices = np.linspace(0, T - 1, num_steps_to_plot, dtype=int)

    ######## Agent 세로로 ##########

    # # ✅ figsize 넉넉하게 조정
    # fig, axes = plt.subplots(A, num_steps_to_plot, figsize=(num_steps_to_plot * 4, A * 2))
    # if A == 1:
    #     axes = np.expand_dims(axes, 0)
    # if num_steps_to_plot == 1:
    #     axes = np.expand_dims(axes, 1)

    ######## Agent 가로로 #############
    fig, axes = plt.subplots(
        num_steps_to_plot, A, figsize=(A * 2.5, num_steps_to_plot * 2.5)
    )

    if num_steps_to_plot == 1:
        axes = np.expand_dims(axes, 0)  # step=1이면 행 추가
    if A == 1:
        axes = np.expand_dims(axes, 1)  # agent=1이면 열 추가

    boundaries = [0, 1, 1 + n_enemy, 1 + n_enemy + n_ally, H]
    labels = ["own", "enemy", "ally", "history"]

    tmax = th.max(first_dead).item()
    vmin = 0  # np.min(np.mean(maps[:tmax], axis=0))
    # vmax = 0.4 #np.max(np.mean(maps[:tmax], axis=0))

    for agent, t in enumerate(first_dead):
        # ax = axes[agent][0] ## agent 세로로
        ax = axes[0, agent]  ## agent 가로로
        mean_heatmap = maps[:t, agent]
        heatmap = np.mean(mean_heatmap, axis=0)

        im = ax.imshow(
            heatmap,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            interpolation="nearest",
            aspect="auto",
        )

        # 기본 토큰 간 grid
        ax.set_xticks(np.arange(W + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(H + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xticks([])
        ax.set_yticks([])

        # 빨간 경계선
        for b in boundaries[1:-1]:
            ax.axvline(b - 0.5, color="red", linewidth=1.5)
            ax.axhline(b - 0.5, color="red", linewidth=1.5)

        # 라벨
        for idx in range(len(labels)):
            start = boundaries[idx]
            end = boundaries[idx + 1]
            center = (start + end - 1) / 2
            ax.text(
                center,
                H + 0.2,
                labels[idx],
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
                transform=ax.transData,
            )

        # if agent == 0:
        #     ax.set_title(f"t={t}")
        ax.set_ylabel(f"Agent {agent}", rotation=90, fontsize=10)

    # ✅ colorbar 위치 조정 (왼쪽으로 이동)
    # cbar_ax = fig.add_axes([0.87, 0.15, 0.015, 0.7]) ### agent 세로
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Attention weight")

    plt.suptitle(f"Attention (Batch {batch_idx})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # ✅ 여유 공간 확보

    if getattr(main_args, "high_hidden_dropout", False):
        fig_name = (
            "Attention_Heatmap_Mean/"
            + main_args.name
            + "_"
            + str(main_args.token_dropout)
            + "_HD"
            + "/"
            + str(layer)
            + "st-layer"
        )
    else:
        fig_name = (
            "Attention_Heatmap_Mean/"
            + main_args.name
            + "_"
            + str(main_args.token_dropout)
            + "/"
            + str(layer)
            + "st-layer"
        )
    save_dir = os.path.join(os.getcwd(), fig_name)
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{task}_batch_{batch_idx}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def init_tasks(task_list, main_args, logger):
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


def train_sequential(
    train_tasks,
    main_args,
    logger,
    learner,
    task2args,
    task2runner,
    task2offlinedata,
    num_filtered_ds,
    t_start=0,
    test_task2offlinedata=None,
):
    ########## start training ##########
    t_env = t_start
    episode = 0  # episode does not matter
    t_max = main_args.t_max
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

    if getattr(main_args, "s0_filter", False):
        task_sizes = np.array([num_filtered_ds[t] for t in train_tasks], dtype=np.float32)
        task_probs = task_sizes / task_sizes.sum()
        wandb.config.update({
                f"num_filtered_ds/{task}": size for task, size in num_filtered_ds.items()
            })
        wandb.log({
        **{f"num_filtered_ds/{task}": size for task, size in num_filtered_ds.items()},
        "time step": 0,  # optional, 초기 step 기준
        })
        # print(num_filtered_ds)

    while t_env < t_max: # while cur_t < t_max:

        if getattr(main_args, "s0_filter", False):
            task = np.random.choice(train_tasks, p=task_probs)
            episode_sample = task2offlinedata[task].sample(batch_size_train)
            draw_sample = False
            if draw_sample:
                all_ds = task2offlinedata[task].sample(num_filtered_ds[task])
                
                ###### state ######
                s0 = all_ds["state"][:, 0,:]
                # 1️⃣ Convert each state into a rounded, hashable tuple
                state_strs = [tuple(np.round(s, 6)) for s in s0.cpu()]

                # 2️⃣ Compute unique states and their occurrence counts
                unique_states, counts = np.unique(state_strs, axis=0, return_counts=True)

                # 3️⃣ Count how many unique states appear N times
                unique_counts, freq = np.unique(counts, return_counts=True)

                # 4️⃣ Plot the histogram
                plt.figure(figsize=(8, 5))
                plt.bar(unique_counts, freq, width=0.6, color='skyblue', edgecolor='black')
                plt.xlabel("Number of occurrences per unique state", fontsize=12)
                plt.ylabel("Number of unique states", fontsize=12)
                plt.title("Distribution of State Duplications", fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)

                # 5️⃣ Save the figure
                plt.tight_layout()
                plt.savefig(f"images/state_duplication_histogram_{task}.png", dpi=300)
                plt.show()
                
                ##### return ######
                rewards = all_ds["reward"]
                discounts = 0.99 ** np.arange(rewards.shape[1])
                returns = (rewards.cpu().squeeze(-1) * discounts).sum(axis=1)  # shape: (2000,)
                
                rounded_returns = np.round(returns, 3)
                unique_returns, counts = np.unique(rounded_returns, return_counts=True)

                # 3️⃣ 중복 빈도 분포 계산 (예: return이 n번 등장한 경우 몇 개?)
                unique_counts, freq = np.unique(counts, return_counts=True)

                # 4️⃣ 그래프 시각화
                plt.figure(figsize=(8, 5))
                plt.bar(unique_counts, freq, width=0.6, color='lightcoral', edgecolor='black')
                plt.xlabel("Number of occurrences per unique return", fontsize=12)
                plt.ylabel("Number of unique returns", fontsize=12)
                plt.title("Distribution of Return Duplications", fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"images/return_duplication_histogram_{task}.png", dpi=300)
                
                exit()
            
            if episode_sample.device != task2args[task].device:
                episode_sample.to(task2args[task].device)

            if callable(update_fn):
                if main_args.split_ds:
                    terminated = learner.train_double(
                    episode_sample, t_env / len(train_tasks), episode, task
                    )
                else:
                    terminated = learner.train(
                        episode_sample, t_env / len(train_tasks), episode, task
                    )
            else:
                if main_args.split_ds:
                    terminated = learner.train_double(episode_sample, t_env, episode, task)
                else:
                    terminated = learner.train(episode_sample, t_env, episode, task)

            if terminated is not None and terminated:
                break

            episode += batch_size_run
            t_env += 1
            # cur_t += 1

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
                    for test_task in main_args.test_tasks:
                        task2runner[test_task].t_env = t_env
                        for _ in range(n_test_runs):
                            task2runner[test_task].run(test_mode=True)

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
                save_path = os.path.join(main_args.save_dir, str(t_env))
                os.makedirs(save_path, exist_ok=True)
                logger.console_logger.info("Saving models to {}".format(save_path))
                learner.save_models(save_path)
                model_save_time = t_env

            if (t_env - last_log_T) >= main_args.log_interval:
                last_log_T = t_env
                logger.log_stat("episode", episode, t_env)
                logger.print_recent_stats()
                max_log_len = max([len(v) for k, v in logger.stats.items()])

                wandb.log(
                    {
                        "time step": t_env,
                        **{
                            f"{k}": v[-1][1]
                            for k, v in logger.stats.items()
                            if len(v) == max_log_len
                        },
                    }
                )


        # cur_t = 0
        
        else:
            np.random.shuffle(train_tasks)
            for task in train_tasks:
            # train each task
                episode_sample = task2offlinedata[task].sample(batch_size_train)
                draw_sample = False
                if draw_sample:
                    all_ds = task2offlinedata[task].sample(2000)
                    
                    ###### state ######
                    state = all_ds["state"].cpu()

                    B, T, D = state.shape
                    device = state.device

                    # --- s_t, s_{t+1} 추출 ---
                    s_t = state[:, :-1, :].reshape(-1, D)      # [B*(T-1), D]
                    s_next = state[:, 1:, :].reshape(-1, D)    # [B*(T-1), D]

                    # --- 근사 비교를 위해 rounding ---
                    precision = 3
                    s_t_rounded = th.round(s_t * 10**precision).int().cpu().numpy()
                    s_next_rounded = th.round(s_next * 10**precision).int().cpu().numpy()

                    # --- transition mapping 구성 ---
                    keys = [tuple(x) for x in s_t_rounded]
                    values = [tuple(x) for x in s_next_rounded]

                    next_dict = defaultdict(set)
                    for k, v in zip(keys, values):
                        next_dict[k].add(v)

                    transition_variety = {k: len(v) for k, v in next_dict.items()}
                    avg_branching = np.mean(list(transition_variety.values()))

                    # --- 다양성 높은 top-5 ---
                    sorted_items = sorted(transition_variety.items(), key=lambda x: -x[1])[:5]

                    branch_counts = [cnt for cnt in transition_variety.values() if cnt > 1]
                    plt.figure(figsize=(6,5))
                    plt.hist(branch_counts, bins=30, color="steelblue", edgecolor="black")
                    plt.xlabel("# of distinct next states per s_t ( > 1 only )")
                    plt.ylabel("Count")
                    plt.title("Transition Diversity Histogram (branching states only)")
                    plt.tight_layout()
                    plt.savefig(f"images/transition_diversity_{task}.png", dpi=300)

###################################################################################################################

                    # # --- 텍스트 파일로 저장 ---
                    # output_path = f"{task}_transition_analysis.txt"
                    # with open(output_path, "w", encoding="utf-8") as f:
                    #     f.write("==== Transition Diversity Analysis ====\n")
                    #     f.write(f"Total unique s_t: {len(transition_variety)}\n")
                    #     f.write(f"Average branching count: {avg_branching:.3f}\n\n")

                    #     f.write("---- Top-5 states with most diverse next states ----\n")
                    #     for i, (k, cnt) in enumerate(sorted_items, 1):
                    #         f.write(f"{i}) s_t={np.array(k)} → {cnt} unique s_(t+1)\n")

                    #     # f.write("\n---- Full transition diversity list ----\n")
                    #     # for k, cnt in transition_variety.items():
                    #     #     f.write(f"s_t={np.array(k)} → {cnt} unique s_(t+1)\n")

                    # print(f"✅ Transition analysis results saved to: {output_path}")
                                        
                    # traj = state.reshape(B, -1)
                    
                    # with th.no_grad():
                    #     dist = th.cdist(traj, traj, p=2)

                    # dist_np = dist.cpu().numpy()
                    # plt.figure(figsize=(6, 5))
                    # im = plt.imshow(dist_np, interpolation='nearest')
                    # plt.title("Trajectory Pairwise Distance")
                    # plt.xlabel("Trajectory index")
                    # plt.ylabel("Trajectory index")
                    # plt.colorbar(im, label="L2 distance")
                    # plt.tight_layout()
                    # plt.savefig(f"images/heatmap_{task}.png", dpi=300)

                    # traj_np = traj.detach().cpu().numpy()
                    # tsne = TSNE(
                    #     n_components=2,
                    #     perplexity=min(30, B-1),  # batch_size 작으면 자동 조절
                    #     metric="euclidean",
                    #     init="pca",
                    #     learning_rate="auto",
                    #     max_iter=1000,
                    #     verbose=1,
                    # )

                    # emb_2d = tsne.fit_transform(traj_np)  # [B, 2]                    
                    # plt.figure(figsize=(6, 5))
                    # plt.scatter(emb_2d[:, 0], emb_2d[:, 1])
                    # plt.title("t-SNE of Trajectories")
                    # plt.xlabel("Dim 1")
                    # plt.ylabel("Dim 2")
                    # plt.tight_layout()
                    # plt.savefig(f"images/tsne_{task}.png", dpi=300)

                    # threshold = 20 # 상황에 맞게 튜닝
                    # similar_counts = (dist < threshold).sum(dim=1) - 1  # 자기 자신 제외 → [B]

                    # plt.figure(figsize=(6, 5))
                    # sc = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=similar_counts.cpu(), s=40)
                    # plt.colorbar(sc, label=f"# of neighbors (dist < {threshold})")
                    # plt.title("t-SNE (colored by # of similar trajectories)")
                    # plt.xlabel("Dim 1")
                    # plt.ylabel("Dim 2")
                    # plt.tight_layout()
                    # plt.savefig(f"images/neighbor_tsne_{task}.png", dpi=300)

                    exit()
                
                if episode_sample.device != task2args[task].device:
                    episode_sample.to(task2args[task].device)

                if callable(update_fn):
                    if main_args.split_ds:
                        terminated = learner.train_double(
                        episode_sample, t_env / len(train_tasks), episode, task
                        )
                    else:
                        terminated = learner.train(
                            episode_sample, t_env / len(train_tasks), episode, task
                        )
                else:
                    if main_args.split_ds:
                        terminated = learner.train_double(episode_sample, t_env, episode, task)
                    else:
                        terminated = learner.train(episode_sample, t_env, episode, task)

                if terminated is not None and terminated:
                    break

                episode += batch_size_run
                t_env += 1
                # cur_t += 1

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
                        for test_task in main_args.test_tasks:
                            task2runner[test_task].t_env = t_env
                            for _ in range(n_test_runs):
                                task2runner[test_task].run(test_mode=True)

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
                    save_path = os.path.join(main_args.save_dir, str(t_env))
                    os.makedirs(save_path, exist_ok=True)
                    logger.console_logger.info("Saving models to {}".format(save_path))
                    learner.save_models(save_path)
                    model_save_time = t_env

                if (t_env - last_log_T) >= main_args.log_interval:
                    last_log_T = t_env
                    logger.log_stat("episode", episode, t_env)
                    logger.print_recent_stats()
                    max_log_len = max([len(v) for k, v in logger.stats.items()])

                    wandb.log(
                        {
                            "time step": t_env,
                            **{
                                f"{k}": v[-1][1]
                                for k, v in logger.stats.items()
                                if len(v) == max_log_len
                            },
                        }
                    )

            # cur_t = 0

def run_sequential(args, logger):
    # Init runner so we can get env info
    args.n_tasks = len(args.train_tasks)
    # define main_args
    main_args = copy.deepcopy(args)

    all_tasks = list(set(args.train_tasks + args.test_tasks))

    task2args, task2runner, task2buffer, task2scheme, task2groups, task2preprocess = (
        init_tasks(all_tasks, main_args, logger)
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
        if main_args.attention_heatmap:
            task2offlinedata = {}
            for task in main_args.train_tasks:
                task2offlinedata[task] = OfflineBuffer(
                    task,
                    main_args.train_tasks_data_quality[task],
                    data_folder=main_args.offline_data_name,
                    offline_data_size=args.offline_data_size,
                    random_sample=args.offline_data_shuffle,
                )
            train_sequential(
                main_args.train_tasks,
                main_args,
                logger,
                learner,
                task2args,
                task2runner,
                task2offlinedata,
            )
            return

    
    # initialize training data for each task
    task2offlinedata = {}
    num_filtered_ds = {}
    if not getattr(args, "s0_filter", False):
        s0_filter_threshold=None
        s0_filter_topk=None
    else:
        s0_filter_threshold=args.s0_filter_threshold
        s0_filter_topk=args.s0_filter_topk
        
    
    for task in main_args.train_tasks:
        # create offline data buffer
        task2offlinedata[task] = OfflineBuffer(
            task,
            main_args.train_tasks_data_quality[task],
            data_folder=main_args.offline_data_name,
            offline_data_size=args.offline_data_size,
            random_sample=args.offline_data_shuffle,
            s0_filter_threshold=s0_filter_threshold, 
            s0_filter_topk=s0_filter_topk
        )
        num_filtered_ds[task] = task2offlinedata[task].buffer.num_data
        
        
        

    logger.console_logger.info(
        "Beginning multi-task offline training with {} timesteps for each task".format(
            main_args.t_max
        )
    )
    train_sequential(
        main_args.train_tasks,
        main_args,
        logger,
        learner,
        task2args,
        task2runner,
        task2offlinedata,
        num_filtered_ds
    )
    wandb.finish()
    # save the final model
    if main_args.save_model:
        save_path = os.path.join(main_args.save_dir, str(main_args.t_max))
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
