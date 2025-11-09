import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from collections import deque
from threading import Lock

from config.drl_config import ddqn_config
from config.config import SEED_GLOBAL, DEVICE

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
    global_memory = deque(maxlen=ddqn_config['maxlen_mem'])

    def __init__(self, state_size, action_size, checkpoint_path="./", load_model=False):
        """
        Khởi tạo một agent DDQN (Deep Double Q-Network) dùng cho Reinforcement Learning.

        Thuộc tính:
            state_size (int): Kích thước vector trạng thái quan sát.
            action_size (int): Số lượng hành động có thể thực hiện.
            checkpoint_path (str): Đường dẫn lưu hoặc load mô hình.
            load_model (bool): Nếu True, tải mô hình đã lưu và đặt epsilon = 0.

        Các thuộc tính chính được thiết lập:
            state_size (int): Lưu state_size.
            action_size (int): Lưu action_size.
            discount_factor (float): Hệ số chiết khấu gamma.
            learning_rate (float): Tốc độ học của mạng neural.
            epsilon (float): Giá trị epsilon cho epsilon-greedy.
            epsilon_decay (float): Tốc độ giảm epsilon theo thời gian.
            epsilon_min (float): Giá trị epsilon tối thiểu.
            batch_size (int): Kích thước batch để huấn luyện.
            train_start (int): Số lượng kinh nghiệm tối thiểu trước khi huấn luyện.
            memory (deque): Bộ nhớ replay riêng của agent.
            global_memory (deque): Bộ nhớ replay dùng chung cho tất cả agent.
            model (nn.Module): Mạng Q chính (online network).
            target_model (nn.Module): Mạng Q mục tiêu (target network).
            criterion: Hàm mất mát MSE.
            optimizer: AdamW optimizer.
            scheduler: ReduceLROnPlateau scheduler cho learning rate.
            generator: Bộ sinh số ngẫu nhiên.
            model_file (str): Đường dẫn lưu/trích xuất mô hình.
            lock: Khóa để đồng bộ thao tác đa luồng.
        """
        super(DDQNAgent, self).__init__()
        self.__load_model = load_model

        self.__state_size = state_size
        self.__action_size = action_size

        self.__discount_factor = ddqn_config["discount_factor"]
        self.__learning_rate = ddqn_config["learning_rate"]
        self.__epsilon = ddqn_config["epsilon"]
        self.__epsilon_decay = ddqn_config["epsilon_decay"]
        self.__epsilon_min = ddqn_config["epsilon_min"]
        self.__batch_size = ddqn_config["batch_size"]
        self.__train_start = self.__batch_size

        # Bộ nhớ agent
        self.__memory = deque(maxlen=ddqn_config["maxlen_mem"])
        self.__global_memory = DDQNAgent.global_memory

        # Mạng Q
        self.__model = self.build_model().to(device)
        self.__target_model = self.build_model().to(device)
        self.__criterion = nn.MSELoss()
        self.__optimizer = optim.AdamW(
            self.__model.parameters(), lr=self.__learning_rate
        )
        self.__scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.__optimizer, mode="min", factor=0.1, patience=5
        )

        # Lưu / load mô hình
        self.__model_file = checkpoint_path
        self.__random = np.random.default_rng(SEED_GLOBAL)

        if self.__load_model:
            self.__model.load_state_dict(torch.load(self.__model_file))
            self.__epsilon = 0

        # Đồng bộ target_model với model
        self.update_target_model()

        # Khóa để thread-safe
        self.__lock = Lock()

    def build_model(self):
        """
        Xây dựng mạng neural cho agent DDQN.

        Kiến trúc:
            - Input: vector trạng thái (state_size)
            - Hidden layers: 3 lớp với số neuron tỉ lệ với state_size
            - Output: vector Q-value cho mỗi hành động (action_size)
            - Kích hoạt: SELU cho 2 lớp đầu, ELU cho lớp thứ 3 và output

        Returns:
            nn.Sequential: Mạng neural hoàn chỉnh.
        """
        # Kích thước các lớp ẩn
        hidden_size_1 = self.__state_size + int(self.__state_size * 0.3)  # 130% state_size
        hidden_size_2 = int(self.__state_size * 0.6)                   # 60% state_size
        hidden_size_3 = int(self.__state_size * 0.2)                   # 20% state_size

        # Xây dựng mạng neural
        model = nn.Sequential(
            nn.Linear(self.__state_size, hidden_size_1),
            nn.SELU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.SELU(),
            nn.Linear(hidden_size_2, hidden_size_3),
            nn.ELU(),
            nn.Linear(hidden_size_3, self.__action_size),
            nn.ELU()
        )

        # Softmax dùng khi cần chuyển Q-values sang xác suất
        self.__softmax = nn.Softmax(dim=1)

        return model
    
    def get_train_start(self):
        """Trả về Số lượng kinh nghiệm tối thiểu trước khi huấn luyện."""
        return self.__train_start
    
    def forward(self, state):
        """Tính Q-values cho trạng thái đầu vào qua mạng chính."""
        q_values = self.__model(state)
        return q_values


    def save_model(self, file_path):
        """Lưu trọng số mạng chính vào file chỉ định."""
        torch.save(self.__model.state_dict(), file_path)


    def update_target_model(self):
        """Cập nhật trọng số của target network từ mạng chính."""
        self.__target_model.load_state_dict(self.__model.state_dict())

    def get_action(self, state, agent_idx):
        """Chọn hành động dựa trên epsilon-greedy policy cho agent."""
        if self.__epsilon > self.__random.random():
            # Chọn hành động ngẫu nhiên
            return np.random.randint(0, self.__action_size)
        else:
            # Chọn hành động tối ưu từ mạng Q
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                q_values = self.__model(state_tensor)
            return torch.argmax(q_values).item()
        
    def get_actions(self, state, vehicle_id):
        """Sinh hành động từ trạng thái hiện tại cho một phương tiện."""
        state = state.to(device)
        with torch.no_grad():
            actions = self(state)
        actions = actions.cpu().detach()
        actions = [vehicle_id, actions]
        return actions, None



    def add_memory(self, state, action, reward, next_state, done=0):
        """Thêm một trải nghiệm vào bộ nhớ của agent."""
        if action == -1:  # Bỏ qua hành động không hợp lệ
            return

        # Giữ kích thước bộ nhớ không vượt quá maxlen
        if len(self.__memory) >= ddqn_config['maxlen_mem']:
            self.__memory.popleft()
        
        # Thêm trải nghiệm vào memory
        self.__memory.append((state, action, reward, next_state, done))


    def add_global_memory(self, state, action, reward, next_state, done=0):
        """Thêm một trải nghiệm vào bộ nhớ toàn cục của agent."""
        if action == -1:  # Bỏ qua hành động không hợp lệ
            return

        # Giữ kích thước bộ nhớ toàn cục không vượt quá maxlen
        if len(self.__global_memory) >= ddqn_config['maxlen_mem']:
            self.__global_memory.popleft()
        
        # Thêm trải nghiệm vào global_memory
        self.__global_memory.append((state, action, reward, next_state, done))


    def train_model(self):
        """
        Huấn luyện model DDQN dựa trên mini-batch từ bộ nhớ (memory hoặc global_memory).
        """
        if eval:
            return

        # Cập nhật epsilon theo decay
        if self.__epsilon > self.__epsilon_min:
            self.__epsilon *= self.__epsilon_decay
            print(f"self.learning_rate {self.__learning_rate}")
        print(f"epsilon = {self.__epsilon}")

        self.__lock.acquire()
        try:
            # Chọn mini-batch từ global_memory hoặc memory
            if self.__random.random() > 1 - ddqn_config['combine'] and len(self.__global_memory) >= self.batch_size:
                mini_batch = random.sample(self.__global_memory, self.__batch_size)
            else:
                mini_batch = random.sample(self.__memory, self.__batch_size)
            
            # Khởi tạo các tensor cho states, next_states, actions, rewards, dones
            states = np.zeros((self.__batch_size, self.__state_size))
            next_states = np.zeros((self.__batch_size, self.__state_size))
            actions, rewards, dones = [], [], []

            # Chuẩn hóa dữ liệu từ mini-batch
            for i in range(self.__batch_size):
                states[i] = mini_batch[i][0]
                actions.append(mini_batch[i][1])
                rewards.append(mini_batch[i][2])
                next_states[i] = mini_batch[i][3]
                dones.append(mini_batch[i][4])

            # Chuyển dữ liệu sang tensor
            states = torch.FloatTensor(states).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(device)  # Tensor 2D
            rewards = torch.FloatTensor(rewards).to(device).reshape([self.__batch_size])
            dones = torch.FloatTensor(dones).to(device)

            # Lấy Q-values từ model chính và target model
            q_values = self.__model(states)
            next_q_values = self.__target_model(next_states).detach()

            # Tính Q-target theo công thức DDQN: Q_target = reward + gamma * max(Q_next)
            max_next_q_values = next_q_values.max(dim=1)[0]
            target_q_values = rewards + (1 - dones) * self.__discount_factor * max_next_q_values

            # Kiểm tra hành động hợp lệ
            assert actions.max().item() < q_values.shape[1], "actions contains invalid indices!"
            assert actions.min().item() >= 0, "actions contains negative indices!"

            # Lấy giá trị Q hiện tại theo hành động đã chọn
            current_q_values = q_values.gather(1, actions).squeeze(1)

            # Huấn luyện: tính loss và backprop
            self.__optimizer.zero_grad()
            loss = self.__criterion(current_q_values, target_q_values)
            loss.backward()
            self.__optimizer.step()
        finally:
            self.__lock.release()

    def quantile_huber_loss(self, y_true, y_pred):
        """
        Tính Quantile Huber Loss giữa giá trị thực và giá trị dự đoán.
        """
        # Tạo các quantile từ 0 đến 1 dựa trên số hành động
        quantiles = torch.linspace(
            1 / (2 * self.__action_size),
            1 - 1 / (2 * self.__action_size),
            self.__action_size
        ).to(device)
        
        batch_size = y_pred.size(0)
        
        # Nhân batch_size lần để so sánh với từng quantile
        tau = quantiles.repeat(batch_size, 1)
        
        # Sai số giữa giá trị thực và giá trị dự đoán
        error = y_true - y_pred
        
        # Huber loss: dùng bình phương nếu nhỏ, tuyến tính nếu lớn
        huber_loss = torch.where(torch.abs(error) < 0.5, 0.5 * error ** 2, torch.abs(error) - 0.5)
        
        # Tính quantile loss: trọng số tau dựa trên dấu của error
        quantile_loss = torch.abs(tau - (error < 0).float()) * huber_loss
        
        # Trả về trung bình loss trên tất cả batch và quantiles
        return quantile_loss.mean()

