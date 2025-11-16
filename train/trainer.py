import numpy as np
import os
import torch
import logging

from core.task_generator import TaskGenerator
from drl.agents.ddqn_agent import DDQNAgent
from drl.envs.data_loader import DataLoader
from drl.envs.environment import Environment
from drl.trainer.ddqn_trainer import DDQNTrainer
from config.config import DEVICE
from config.drl_config import ddqn_config, epoch_size
from ray.tune.registry import register_env

logging.basicConfig(
    level=logging.INFO,  # có thể đổi thành INFO khi muốn giảm log, DEBUG để hiển thị toàn bộ log
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("./logs/app.log", mode="a", encoding="utf-8"),
        logging.StreamHandler(),  # In ra màn hình
    ],
)

logger = logging.getLogger(__name__)

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
        state_dim=state_dim,
        action_dim=action_dim,
        model_path=checkpoint_path,
        load_pretrained=load_model,
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

    # Khởi tạo trainer với các tham số cấu hình
    trainer = DDQNTrainer(
        env=environment,
        agents=agent_list,
        score_window_size=score_window_size,
        max_episode_length=max_episode_length,
        update_frequency=update_interval,
        save_dir=save_directory,
        use_thread=use_thread,
        detach_thread=detach_thread,
        train_start_factor=2,
    )

    # Khởi tạo bộ đếm
    trainer.current_step = 0
    trainer.current_episode = 0

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
    logger.info("===== BẮT ĐẦU HUẤN LUYỆN AGENTS =====")
    logger.info(f"Max episodes: {max_episodes}")
    logger.info(f"Target score: {target_score}")
    logger.info(f"Score window: {score_window}")
    logger.info(f"Số agents: {len(trainer.agents)}")
    logger.info(f"Môi trường: {env.__class__.__name__}")
    logger.info(f"Trainer: {trainer.__class__.__name__}")

    for episode_idx in range(1, max_episodes + 1):
        try:
            logger.debug(f"[Episode {episode_idx}] Bắt đầu episode...")
            # Thực hiện 1 bước huấn luyện (episode)
            trainer.run_episode_step()

            # In trạng thái huấn luyện định kỳ
            if episode_idx % 100 == 0:
                trainer.print_status()

            # Tính điểm trung bình của các episode gần nhất
            recent_scores = trainer.score_history[-score_window:]
            logger.debug(
                f"[Episode {episode_idx}] Score history length: {len(trainer.score_history)}"
            )
            mean_reward = np.max(recent_scores, axis=1).mean()
            logger.info(
                f"Episode {episode_idx} - Mean reward (last {score_window} episodes): {mean_reward:.2f}"
            )

            logger.debug(
                f"[Episode {episode_idx}] Mean reward computed from max rewards per episode."
            )

            # Lưu model và plot định kỳ
            if episode_idx % epoch_size == 0:
                logger.debug(f"[Episode {episode_idx}] Lưu model và plot định kỳ.")
                trainer.save_models()
                trainer.print_status()
                trainer.df_scores()
            elif episode_idx % score_window == 0:
                logger.debug(
                    f"[Episode {episode_idx}] Cập nhật df_scores() theo score_window."
                )
                trainer.print_status()
                trainer.df_scores()

            # Dừng huấn luyện nếu đạt target_score hoặc hết max_episodes
            if mean_reward >= target_score:
                logger.info(
                    f"⛳ Target đạt được! Mean reward = {mean_reward:.2f} >= {target_score}"
                )
                logger.debug("Bắt đầu lưu model cuối cùng trước khi thoát.")
                trainer.save_models()
                trainer.print_status()
                trainer.df_scores()
                logger.debug("Đóng môi trường.")
                env.close()
                break

            if episode_idx == max_episodes:
                logger.info("🛑 Đã đạt max_episodes, dừng huấn luyện.")
                trainer.save_models()
                trainer.print_status()
                trainer.df_scores()
                logger.debug("Đóng môi trường.")
                env.close()
                break
        except Exception as e:
            logger.error(f"[ERROR] Running error: {e}")

# ddqn
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
            env_data=config, verbose=verbose, map_obj=map_info, task_generator=task_gen
        ),
    )

    env = Environment(
        env_data=env_config, verbose=verbose, map_obj=map_info, task_generator=task_gen
    )

    # --- 5. Extract environment dimensions ---
    num_agents = env_config["num_vehicles"]
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.shape[0]

    logger.info(
        f"====== num_agents: {num_agents}, state_dim: {state_dim}, action_dim: {action_dim}"
    )

    # --- 6. Initialize DDQN agents ---
    agents = []
    for i in range(num_agents):
        agent = create_agent(
            state_dim, action_dim, agent_idx=i, load_model=False, checkpoint_idx=0
        )
        agents.append(agent)

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

    # --- 8. Create trainer ---
    trainer = create_trainer(
        env,
        agents,
        save_dir,
        use_thread=env_config["apply_thread"],
        detach_thread=env_config["apply_detach"],
        score_window_size=env_config["score_window_size"],
        max_episode_length=env_config["max_missions_per_vehicle"] * env_config["num_vehicles"],
        update_interval=ddqn_config["batch_size"] / 4,
    )

    # --- 9. Train agents ---
    train_agents(env, trainer, score_window=env_config["score_window_size"])


if __name__ == "__main__":
    run_ddqn_training()
