import numpy as np
import os
import torch

from core.task_generator import TaskGenerator
from drl.agents.ddqn_agent import DDQNAgent
from drl.envs.data_loader import DataLoader
from drl.envs.enviroment import Environment
from drl.trainer.ddqn_trainer import DDQNTrainer
from config.config import DEVICE
from config.drl_config import ddqn_config
from ray.tune.registry import register_env

if DEVICE != "cpu":
    device = torch.device("cuda:" + str(DEVICE) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
if device == "cpu":
    print("cannot train with cpu")
    exit(0)
else:
    print("cuda: ", device)


def create_agent(
    state_dim,
    action_dim,
    agent_idx=0,
    load_model=False,
    checkpoint_idx=0,
):
    """
    Tạo một agent (DDQN) với các tham số định nghĩa.

    Args:
        state_dim (int): Số chiều trạng thái đầu vào.
        action_dim (int): Số chiều hành động.
        agent_idx (int): Chỉ số của agent (dùng cho checkpoint).
        load_model (bool): Nếu True, load model từ file.
        checkpoint_idx (int): Số thứ tự checkpoint nếu muốn load nhiều version.

    Returns:
        agent: Một object agent (DDQNAgent) đã khởi tạo.
    """

    # Xác định đường dẫn checkpoint
    if checkpoint_idx == 0:
        checkpoint_path = f"./checkpoints/agent_{agent_idx}.pth"
    else:
        checkpoint_path = f"./checkpoints/agent_{agent_idx}_{checkpoint_idx}.pth"

    # Khởi tạo Agent
    agent = DDQNAgent(
        state_size=state_dim,
        action_size=action_dim,
        checkpoint_path=checkpoint_path,
        load_model=load_model,
    )

    return agent


def create_trainer(
    environment,
    agent_list,
    save_directory,
    update_interval=500,
    max_episode_length=100,
    score_window_size=100,
    use_thread=True,
    detach_thread=True,
    trainer_type="DDQNTrainer",
):
    """
    Tạo và khởi tạo trainer để huấn luyện các agent trong môi trường.

    Args:
        environment: Môi trường mô phỏng để huấn luyện và đánh giá agent.
        agent_list: Danh sách các agent cần huấn luyện.
        save_directory: Thư mục để lưu mô hình, checkpoint và log.
        update_interval: Số bước giữa các lần cập nhật mạng mục tiêu.
        max_episode_length: Số bước tối đa trong mỗi episode.
        score_window_size: Kích thước cửa sổ để tính điểm trung bình.
        use_thread: Có sử dụng đa luồng trong huấn luyện hay không.
        detach_thread: Có tách luồng khỏi tiến trình chính hay không.
        trainer_type: Loại trainer (mặc định là "DDQNTrainer").

    Returns:
        Trainer: Đối tượng trainer được khởi tạo sẵn sàng để huấn luyện.
    """

    # Đảm bảo thư mục lưu trữ tồn tại
    os.makedirs(save_directory, exist_ok=True)

    # Xác định loại trainer cần khởi tạo
    trainer_class = DDQNTrainer

    # Khởi tạo trainer với các tham số cấu hình
    trainer = trainer_class(
        env=environment,
        agents=agent_list,
        score_window_size=score_window_size,
        max_episode_length=max_episode_length,
        update_frequency=update_interval,
        save_dir=save_directory,
        use_thread=use_thread,
        use_detach_thread=detach_thread,
        train_start_factor=2,
    )

    # Khởi tạo bộ đếm
    trainer.set_time_step(0)
    trainer.set_episode_count(0)

    return trainer


def train_agents(
    env, trainer, max_episodes=100000, target_score=100000, score_window=100
):
    """
    Thực hiện quá trình huấn luyện các agent trong môi trường.

    Args:
        env: Môi trường RL (multi-agent environment).
        trainer: Đối tượng trainer (MAPPOTrainer hoặc DDQNTrainer) quản lý quá trình huấn luyện.
        max_episodes: Số lượng tối đa episode huấn luyện.
        target_score: Mức điểm trung bình tối thiểu để coi là đã "hoàn thành" môi trường.
        score_window: Số episode gần nhất để tính điểm trung bình.
    """

    for episode_idx in range(1, max_episodes + 1):
        # Thực hiện 1 bước huấn luyện (episode)
        trainer.run_episode()

        # In trạng thái huấn luyện định kỳ
        if episode_idx % 100 == 0:
            trainer.print_status()

        # Tính điểm trung bình của các episode gần nhất
        recent_scores = np.array(trainer.get_score_history()[:,-score_window:])
        mean_reward = np.max(recent_scores, axis=1).mean()
        print(
            f"Episode {episode_idx} - Mean reward (last {score_window} episodes): {mean_reward:.2f}"
        )

        # Lưu model và plot định kỳ
        if episode_idx % 1000 == 0:
            trainer.save()
            trainer.print_status()
            trainer.plot()
        elif episode_idx % score_window == 0:
            trainer.print_status()
            trainer.plot()

        # Dừng huấn luyện nếu đạt target_score hoặc hết max_episodes
        if mean_reward >= target_score or episode_idx == max_episodes:
            print("Môi trường đã được giải quyết hoặc đạt max episode.")
            trainer.save()
            trainer.print_status()
            trainer.plot()
            env.close()
            break


def run_ddqn_training(**kwargs):
    """
    Khởi tạo và huấn luyện các agent sử dụng thuật toán DDQN trong môi trường ITS.

    Tham số:
        verbose (bool, optional): Nếu True, in log chi tiết trong quá trình huấn luyện. Mặc định là False.

    Quy trình:
        1. Tải dữ liệu bản đồ và thông tin môi trường.
        2. Tạo bộ sinh tác vụ (TaskGenerator).
        3. Ghi cấu hình môi trường.
        4. Đăng ký và khởi tạo môi trường huấn luyện.
        5. Xác định số lượng agent, kích thước trạng thái và hành động.
        6. Tạo danh sách các agent DDQN.
        7. Cấu hình thư mục lưu kết quả.
        8. Khởi tạo Trainer và bắt đầu huấn luyện.
    """
    verbose = kwargs.get("verbose", True)

    # --- 1. Load environment and map information ---
    data_loader = DataLoader()
    graph, map_info = data_loader.get_graph_and_map()

    # --- 2. Create task generator ---
    task_gen = TaskGenerator(1, map_info)

    # --- 3. Write environment config ---
    env_config = DataLoader.generate_config_not_from_file(mission_generator=task_gen)

    # --- 4. Register and create environment ---
    register_env(
        "env",
        lambda config: Environment(
            data=config, verbose=verbose, map_obj=map_info, task_generator=task_gen
        ),
    )

    env = Environment(
        data=env_config, verbose=verbose, map_obj=map_info, task_generator=task_gen
    )

    # --- 5. Extract environment dimensions ---
    num_agents = env_config["num_vehicles"]
    state_dim = np.prod(env.get_observation_space().shape)
    action_dim = env.get_action_space().shape[0]

    print('-------------------num_agents, state_dim, action_dim-----------------')
    print(num_agents, state_dim, action_dim)

    # --- 6. Initialize DDQN agents ---
    agents = []
    for i in range(num_agents):
        agent = create_agent(
            state_dim, action_dim, agent_idx=i, load_model=False, checkpoint_idx=0
        )
        print(i, agent)
        agents.append(agent)

    print('-------------------agents-----------------')
    print(agents)

    # --- 7. Prepare save directory ---
    save_dir = os.path.join(
        os.getcwd(),
        "saved_files_global_combine_decay_{decay}_lr_{lr}_batch_{bs}_reward_{rw}_combine_{cb}_more".format(
            decay=ddqn_config["epsilon_decay"],
            lr=ddqn_config["learning_rate"],
            bs=ddqn_config["batch_size"],
            rw=ddqn_config["modify_reward"],
            cb=ddqn_config["combine"],
        ),
    )

    print('-------------------save_dir-----------------')
    print(save_dir)

    # --- 8. Create trainer ---
    trainer = create_trainer(
        env,
        agents,
        save_dir,
        use_thread=env_config["apply_thread"],
        detach_thread=env_config["apply_detach"],
        score_window_size=env_config["score_window_size"],
        max_episode_length=env_config["max_missions_per_vehicle"] * env_config["num_vehicles"],
        trainer_type="DDQNTrainer",
        update_interval=ddqn_config["batch_size"] / 4,
    )

    print('-------------------trainer-----------------')
    print(trainer)

    # --- 9. Train agents ---
    train_agents(env, trainer, score_window=env_config["score_window_size"])

if __name__ == '__main__':
    run_ddqn_training()
