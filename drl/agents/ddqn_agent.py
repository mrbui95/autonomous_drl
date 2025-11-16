import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

from collections import deque
from threading import Lock

from config.drl_config import ddqn_config
from config.config import SEED_GLOBAL, DEVICE, eval

import logging

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


class DDQNAgent(nn.Module):
    # Global memory dùng chung cho tất cả instance của DDQNAgent
    global_memory = deque(maxlen=ddqn_config["maxlen_mem"])

    def __init__(self, state_dim, action_dim, model_path="./", load_pretrained=False):
        """
        Khởi tạo một agent Double DQN.

        Args:
            state_dim (int): Kích thước vector trạng thái.
            action_dim (int): Số lượng action khả thi.
            model_path (str): Đường dẫn lưu hoặc load model.
            load_pretrained (bool): Nếu True, sẽ load model đã lưu sẵn.
        """
        super(DDQNAgent, self).__init__()

        # --- Thông số học ---
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount = ddqn_config["discount_factor"]  # Gamma
        self.lr = ddqn_config["learning_rate"]  # Learning rate
        self.epsilon = ddqn_config["epsilon"]  # Khả năng khám phá ban đầu
        self.epsilon_decay = ddqn_config["epsilon_decay"]  # Hệ số decay epsilon
        self.epsilon_min = ddqn_config["epsilon_min"]  # Giá trị epsilon tối thiểu
        self.batch_size = ddqn_config["batch_size"]  # Số mẫu mỗi batch
        self.train_start = self.batch_size  # Bắt đầu train khi memory đủ batch

        # --- Memory ---
        self.memory = deque(maxlen=ddqn_config["maxlen_mem"])  # Memory riêng cho agent
        self.global_memory = (
            DDQNAgent.global_memory
        )  # Memory dùng chung cho tất cả agent

        # --- Mạng Q ---
        self.model = self.build_model().to(device)  # Q-network chính
        self.target_model = self.build_model().to(device)  # Target Q-network
        self.loss_fn = nn.MSELoss()  # Loss function
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)  # Optimizer
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=5
        )

        # --- Path và seed ---
        self.model_path = model_path
        self.rng = np.random.default_rng(SEED_GLOBAL)  # Random generator có seed

        # --- Load model nếu có ---
        if load_pretrained:
            self.model.load_state_dict(torch.load(self.model_path))
            self.epsilon = 0  # Nếu load model, không cần khám phá

        # --- Đồng bộ target network với model chính ---
        self.update_target_model()

        # --- Lock để train đa luồng an toàn ---
        self.lock = Lock()

    def build_model(self):
        """
        Xây dựng mạng neural network cho agent Double DQN.

        Cấu trúc:
            Input layer -> Hidden Layer 1 (SELU) -> Hidden Layer 2 (SELU)
            -> Hidden Layer 3 (ELU) -> Output Layer (ELU)

        Returns:
            model (nn.Sequential): Q-network trả về Q-value cho mỗi action.
        """
        # --- Kích thước các layer ẩn dựa trên state_dim ---
        hidden_dim1 = self.state_dim + int(self.state_dim * 0.3)  # Layer 1: 130%
        hidden_dim2 = int(self.state_dim * 0.6)  # Layer 2: 60%
        hidden_dim3 = int(self.state_dim * 0.2)  # Layer 3: 20%

        # --- Xây dựng model theo nn.Sequential ---
        model = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim1),
            nn.SELU(),  # Activation SELU cho layer 1
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.SELU(),  # Activation SELU cho layer 2
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ELU(),  # Activation ELU cho layer 3
            nn.Linear(hidden_dim3, self.action_dim),
            nn.ELU(),  # Output layer ELU, trả về Q-value
        )

        # --- Lưu Softmax nếu muốn dùng sau ---
        self.softmax = nn.Softmax(dim=1)

        return model

    def forward(self, state_tensor):
        """Forward pass qua mạng Q."""
        q_values = self.model(state_tensor)
        # Nếu muốn dùng Softmax cho action probabilities, bỏ comment dòng dưới
        # q_values = self.softmax(q_values)
        return q_values

    def save_model(self, name):
        """Lưu trọng số model vào file."""
        torch.save(self.model.state_dict(), name)

    def delete_old_model(self, path):
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Đã xóa file model cũ: {path}")
        else:
            logger.debug(f"File model cũ không tồn tại, không cần xóa: {path}")

    def update_target_model(self):
        logger.debug("===== [UPDATE TARGET MODEL] =====")

        with torch.no_grad():
            main_weights = torch.cat([p.data.flatten() for p in self.model.parameters()])
            logger.debug(
                f"[MAIN MODEL] weights: min={main_weights.min():.6f}, "
                f"max={main_weights.max():.6f}, mean={main_weights.mean():.6f}"
            )

        """Cập nhật target network từ model chính."""
        self.target_model.load_state_dict(self.model.state_dict())

        with torch.no_grad():
            target_weights = torch.cat([p.data.flatten() for p in self.target_model.parameters()])
            logger.debug(
                f"[TARGET MODEL] updated: min={target_weights.min():.6f}, "
                f"max={target_weights.max():.6f}, mean={target_weights.mean():.6f}"
            )

    # get_action
    def select_action(self, state, agent_id=None):
        """Chọn action theo epsilon-greedy."""
        # Khám phá ngẫu nhiên
        if self.epsilon > self.rng.random():
            return np.random.randint(0, self.action_dim)

        # Khai thác: chọn action có Q-value lớn nhất
        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    # get_actions
    def select_actions(self, state_tensor, agent_id):
        """
        Trả về Q-values cho tất cả action của agent, kèm ID agent.
        """
        # Chuyển state lên device
        state_tensor = state_tensor.to(device)

        # Forward pass để lấy Q-values
        with torch.no_grad():
            q_values = self(state_tensor)

        # Chuyển về CPU và detach khỏi graph
        q_values = q_values.cpu().detach()

        # Trả về danh sách [agent_id, q_values] và None (placeholder)
        return [agent_id, q_values], None

    # add_memory
    def add_experience(self, state, action, reward, next_state, done=0):
        """
        Thêm experience vào bộ nhớ replay của agent.
        """
        # Bỏ qua action không hợp lệ
        if action == -1:
            return

        # Nếu memory đầy, loại bỏ experience cũ nhất
        if len(self.memory) >= ddqn_config["maxlen_mem"]:
            self.memory.popleft()

        # Thêm experience mới
        self.memory.append((state, action, reward, next_state, done))

    # add_global_memory
    def add_global_experience(self, state, action, reward, next_state, done=0):
        """
        Thêm experience vào bộ nhớ toàn cục (global memory) dùng chung cho tất cả agent.
        """
        # Bỏ qua action không hợp lệ
        if action == -1:
            return

        # Nếu global memory đầy, loại bỏ experience cũ nhất
        if len(self.global_memory) >= ddqn_config["maxlen_mem"]:
            self.global_memory.popleft()

        # Thêm experience mới vào global memory
        self.global_memory.append((state, action, reward, next_state, done))

    def train_model(self):
        """
        Huấn luyện Q-network dựa trên mini-batch từ memory.
        """
        # Nếu đang ở chế độ đánh giá, không train
        if eval:
            logger.info(f"eval ===> do nothing")
            return

        # Cập nhật epsilon decay
        if self.epsilon > self.epsilon_min:
            logger.info(f"[EPSILON-DECAY] epsilon={self.epsilon:.6f}, lr={self.lr}")
            self.epsilon *= self.epsilon_decay
            logger.info(f"Learning rate: {self.lr}")
        logger.debug(f"[TRAIN] Epsilon hiện tại = {self.epsilon:.6f}")

        # Lock để train an toàn khi đa luồng
        self.lock.acquire()
        try:
            # Chọn mini-batch từ global memory hoặc local memory
            if (
                self.rng.random() > 1 - ddqn_config["combine"]
                and len(self.global_memory) >= self.batch_size
            ):
                logger.debug(
                    f"[MEMORY] Sử dụng GLOBAL memory - size={len(self.global_memory)}"
                )
                mini_batch = random.sample(self.global_memory, self.batch_size)
            else:
                logger.debug(f"[MEMORY] Sử dụng LOCAL memory - size={len(self.memory)}")
                mini_batch = random.sample(self.memory, self.batch_size)

            logger.debug(f"[MEMORY] Mini-batch size = {len(mini_batch)}")

            # Khởi tạo arrays cho states, next_states
            states = np.zeros((self.batch_size, self.state_dim))
            next_states = np.zeros((self.batch_size, self.state_dim))
            actions, rewards, dones = [], [], []

            # Chuẩn bị dữ liệu từ mini-batch
            for i in range(self.batch_size):
                states[i] = mini_batch[i][0]
                actions.append(mini_batch[i][1])
                rewards.append(mini_batch[i][2])
                next_states[i] = mini_batch[i][3]
                dones.append(mini_batch[i][4])

            logger.debug(
                f"[BATCH] actions(min={min(actions)}, max={max(actions)}, len={len(actions)})"
            )
            logger.debug(
                f"[BATCH] rewards={rewards}"
            )
            logger.debug(f"[BATCH] dones = {sum(dones)} true / {len(dones)}")

            # Chuyển dữ liệu sang tensor trên device
            states = torch.FloatTensor(states).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(device)  # Tensor 2D
            rewards = torch.FloatTensor(rewards).to(device).reshape([self.batch_size])
            dones = torch.FloatTensor(dones).to(device)

            logger.debug(f"[TENSOR] states.shape = {states.shape}")
            logger.debug(f"[TENSOR] next_states.shape = {next_states.shape}")

            # Tính Q-values hiện tại và Q-values target
            q_values = self.model(states)
            next_q_values = self.target_model(next_states).detach()

            logger.debug(
                f"[Q] q_values.shape={q_values.shape}, next_q_values.shape={next_q_values.shape}"
            )

            # Tính Q-target theo Double DQN
            max_next_q_values = next_q_values.max(dim=1)[0]
            target_q_values = rewards + (1 - dones) * self.discount * max_next_q_values

            logger.debug(
                f"[TARGET] reward_avg={rewards.mean():.3f}, "
                f"max_next_q_avg={max_next_q_values.mean():.3f}, "
                f"target_avg={target_q_values.mean():.3f}"
            )

            # Kiểm tra action hợp lệ
            assert (
                actions.max().item() < q_values.shape[1]
            ), "Action vượt ngoài phạm vi!"
            assert actions.min().item() >= 0, "Action âm không hợp lệ!"

            # Lấy Q-value hiện tại tương ứng với action đã chọn
            current_q_values = q_values.gather(1, actions).squeeze(1)
            logger.debug(
                f"[Q-CURRENT] current_q_avg={current_q_values.mean().item():.3f}"
            )

            # Backpropagation
            self.optimizer.zero_grad()
            loss = self.loss_fn(current_q_values, target_q_values)
            loss.backward()
            self.optimizer.step()

            logger.debug(f"[UPDATE] Loss = {loss.item():.6f}")

        finally:
            self.lock.release()

    def quantile_huber_loss(self, y_true, y_pred):
        """
        Tính Quantile Huber Loss cho distributional RL.
        """
        # Xác định các quantile
        quantiles = torch.linspace(
            1 / (2 * self.action_dim), 1 - 1 / (2 * self.action_dim), self.action_dim
        ).to(device)

        batch_size = y_pred.size(0)
        tau = quantiles.repeat(batch_size, 1)  # Lặp quantiles cho batch

        # Sai số giữa giá trị thực và dự đoán
        error = y_true - y_pred

        # Huber loss
        huber_loss = torch.where(
            torch.abs(error) < 0.5, 0.5 * error**2, torch.abs(error) - 0.5
        )

        # Quantile loss
        quantile_loss = torch.abs(tau - (error < 0).float()) * huber_loss

        return quantile_loss.mean()
