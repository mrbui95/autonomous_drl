import numpy as np
import random
import sys
from ray.tune.registry import register_env

import logging

from core.task_generator import TaskGenerator
from simulation.agents.base.max_profit_agent import MaxProfitAgent
from simulation.agents.drl.ddqn_agent import DDQNAgent
from simulation.envs.data_loader import DataLoader
from simulation.envs.environment import Environment
from train.trainer.ddqn_trainer import DDQNTrainer
from config.config import mission_config

sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,  # có thể đổi thành INFO khi muốn giảm log, DEBUG để hiển thị toàn bộ log
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("./logs/app.log", mode="a", encoding="utf-8"),
        logging.StreamHandler(),  # In ra màn hình
    ],
)

def make_environment(**kwargs):
    verbose = kwargs.get("verbose", True)

    # --- 1. Load environment and map information ---
    data_loader = DataLoader()
    graph, map_info = data_loader.get_graph_and_map()

    task_gen = TaskGenerator(1, map_info)

    env_config = data_loader.generate_config_from_file()

    register_env(
        "env",
        lambda config: Environment(
            env_data=config, verbose=verbose, map_obj=map_info, task_generator=task_gen
        ),
    )

    env = Environment(
        env_data=env_config, verbose=verbose, map_obj=map_info, task_generator=task_gen
    )

    num_agents = env_config["num_vehicles"]

    return env, map_info, num_agents


def evaluate_ddqn(env, num_agents, checkpoint_idx):
    agent_list = []
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.shape[0]
    for agent_idx in range(num_agents):
        checkpoint_path = f"./checkpoints/agent_{agent_idx}_{checkpoint_idx}.pth"
        agent = DDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            model_path=checkpoint_path,
            load_pretrained=True,
        )
        agent_list.append(agent)

    ddqn_trainer = DDQNTrainer(
        env=env,
        agents=agent_list,
        score_window_size=100,
        max_episode_length=20 * mission_config["max_missions_per_vehicle"],
        update_frequency=100,
        save_dir="",
        use_thread=True,
        detach_thread=True,
        train_start_factor=2,
    )

    ddqn_trainer.current_step = 0

    ddqn_trainer.env.reset_environment(predict=True)

    ddqn_trainer.run_episode_step()

    ddqn_trainer.print_status()


def evaluate_max_profit(env):
    mission_list = env.missions
    sorted_missions = sorted(
        mission_list,
        key=lambda m: len(m.get_dependencies())
    )
    vehicle_list = env.vehicles

    vehicle_ids = [v.get_vehicle_id() for v in vehicle_list] * env.env_data['max_missions_per_vehicle']
    random.shuffle(vehicle_ids)

    actions = [None] * len(sorted_missions)

    for idx, mission in enumerate(sorted_missions):
        mission_id = mission.get_mission_id()
        actions[mission_id] = (idx // 5, vehicle_ids[idx])

    env.step_multi_agent(actions, 'MAX_PROFIT')


if __name__ == "__main__":
    num_test = 20
    env, map_info, num_agents = make_environment()

    for i in range(num_test):
        num_episodes = 2 * mission_config['max_missions_per_vehicle']
        # with open(f"./evaluate/drl_result_{i}.txt", "w", encoding="utf-8") as file:
        #     original_stdout = sys.stdout
        #     sys.stdout = file
        evaluate_ddqn(env, num_agents, 84000)
        
        env.reset_environment_meta()
        
        # with open(f"./evaluate/mp_result_{i}.txt", "w", encoding="utf-8") as file:
        #     original_stdout = sys.stdout
        #     sys.stdout = file
        evaluate_max_profit(env)
