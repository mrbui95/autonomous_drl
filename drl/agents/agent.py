import torch
import numpy as np

from agents.replay_memory import ReplayMemory


class Agent:
    """
    Lớp cơ sở cho tác tử (agent) trong học tăng cường.

    Cung cấp khung chung cho việc:
    - Tương tác với môi trường và thu thập kinh nghiệm.
    - Huấn luyện mô hình dựa trên dữ liệu trong bộ nhớ kinh nghiệm.
    - Chọn hành động, cập nhật mạng mục tiêu và đánh giá hiệu năng.
    """

    def __init__(
        self,
        env,
        state_dim,
        action_dim,
        memory_capacity=10000,
        max_steps=10000,
        gamma=0.99,
        reward_scale=1.0,
        done_penalty=None,
        actor_hidden_size=32,
        critic_hidden_size=32,
        actor_output_act=identity,
        critic_loss="mse",
        actor_lr=0.01,
        critic_lr=0.01,
        optimizer_type="rmsprop",
        entropy_reg=0.01,
        max_grad_norm=0.5,
        batch_size=100,
        warmup_episodes=100,
        epsilon_start=0.9,
        epsilon_end=0.01,
        epsilon_decay=200,
        use_cuda=True,
        log_file=None,
    ):

        # Môi trường học tăng cường
        self.__env = env
        self.__state_dim = state_dim
        self.__action_dim = action_dim
        self.__env_state = self.__env.reset()

        # Biến đếm số tập và bước
        self.__episode_count = 0
        self.__step_count = 0
        self.__max_steps = max_steps
        self.__rollout_steps = 1

        # Tham số phần thưởng
        self.__gamma = gamma
        self.__reward_scale = reward_scale
        self.__done_penalty = done_penalty

        # Bộ nhớ lưu kinh nghiệm (Replay Buffer)
        self.__memory = ReplayMemory(memory_capacity)

        # Thông số mạng Actor và Critic
        self.__actor_hidden_size = actor_hidden_size
        self.__critic_hidden_size = critic_hidden_size
        self.__actor_output_act = actor_output_act
        self.__critic_loss = critic_loss
        self.__actor_lr = actor_lr
        self.__critic_lr = critic_lr
        self.__optimizer_type = optimizer_type
        self.__entropy_reg = entropy_reg
        self.__max_grad_norm = max_grad_norm
        self.__batch_size = batch_size
        self.__warmup_episodes = warmup_episodes
        self.__target_tau = 0.01

        # Tham số cho chiến lược epsilon-greedy
        self.__epsilon_start = epsilon_start
        self.__epsilon_end = epsilon_end
        self.__epsilon_decay = epsilon_decay

        # Thiết lập thiết bị (CPU/GPU)
        self.__use_cuda = use_cuda and torch.cuda.is_available()

        # File ghi log (nếu có)
        self.__log_file = log_file
        if self.__log_file:
            with open(self.__log_file, "w"):
                pass

    # ----------------------------------------------------------------------
    # Tương tác với môi trường
    # ----------------------------------------------------------------------

    def interact(self):
        """Tác tử tương tác với môi trường để thu thập dữ liệu (chưa cài đặt)."""
        pass

    def log(self, data):
        """Ghi dữ liệu huấn luyện hoặc đánh giá ra file log."""
        raise NotImplementedError("Chưa cài đặt hàm log!")

    def _take_one_step(self, current_datetime=None, rfile=False):
        """
        Thực hiện một bước tương tác với môi trường.

        - Chọn hành động dựa trên trạng thái hiện tại.
        - Nhận phần thưởng, trạng thái kế tiếp, và cờ dừng.
        - Lưu lại kết quả vào bộ nhớ kinh nghiệm.
        """
        state = self.__env_state
        action = self.exploration_action(state)
        next_state, reward, done, _, _ = self.__env.step(action)

        # Nếu tập đã kết thúc, reset lại môi trường
        if done:
            if self.__done_penalty is not None:
                reward = self.__done_penalty
            next_state = [0] * len(state)
            self.__env_state = self.__env.reset(
                current_datetime=current_datetime, rfile=rfile
            )
            self.__episode_count += 1
            self.__episode_done = True
        else:
            self.__env_state = next_state
            self.__episode_done = False

        self.__step_count += 1
        self.__memory.add(state, action, reward, next_state, done)

    def _take_multiple_steps(self):
        """
        Thực hiện nhiều bước liên tiếp trong môi trường và lưu phần thưởng chiết khấu.
        """
        # Nếu vượt quá số bước tối đa thì reset môi trường
        if (self.__max_steps is not None) and (self.__step_count >= self.__max_steps):
            self.__env_state = self.__env.reset()
            self.__step_count = 0

        states, actions, rewards = [], [], []
        states.append(self.__env_state)

        action = self.exploration_action(self.__env_state)
        next_state, reward, done, _, _ = self.__env.step(action)

        actions.append(action)
        if done and self.__done_penalty is not None:
            reward = self.__done_penalty
        rewards.append(reward)

        self.__env_state = next_state if not done else self.__env.reset()

        # Tính giá trị cuối cùng để chiết khấu phần thưởng
        if done:
            final_value = 0.0
            self.__episode_count += 1
            self.__episode_done = True
        else:
            self.__episode_done = False
            final_action = self.action(next_state)
            final_value = self.value(next_state, final_action)

        discounted_rewards = self._discount_rewards(rewards, final_value)
        self.__step_count += 1
        self.__memory.add(states, actions, discounted_rewards)

    def _discount_rewards(self, rewards, final_value):
        """
        Tính phần thưởng chiết khấu theo hệ số gamma.
        """
        rewards = np.asarray(rewards, dtype=np.float32)

        # Nếu final_value là mảng numpy thì chuyển sang số thực
        if isinstance(final_value, np.ndarray):
            final_value = (
                float(np.mean(final_value))
                if final_value.size > 1
                else final_value.item()
            )

        discounted = np.zeros_like(rewards, dtype=np.float32)
        running_total = float(final_value)

        # Duyệt ngược để chiết khấu phần thưởng
        for t in reversed(range(len(rewards))):
            running_total = running_total * self.__gamma + rewards[t]
            discounted[t] = running_total
        return discounted

    # ----------------------------------------------------------------------
    # Huấn luyện
    # ----------------------------------------------------------------------

    def _soft_update_target(self, target, source):
        """
        Cập nhật mềm tham số của mạng mục tiêu.
        θ_target ← (1 - τ) * θ_target + τ * θ_source
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                (1.0 - self.__target_tau) * target_param.data
                + self.__target_tau * source_param.data
            )

    def train(self):
        """Huấn luyện mô hình dựa trên dữ liệu trong Replay Memory (chưa cài đặt)."""
        pass

    # ----------------------------------------------------------------------
    # Chọn hành động
    # ----------------------------------------------------------------------

    def exploration_action(self, state):
        """Chọn hành động có thêm nhiễu để khám phá (trong giai đoạn huấn luyện)."""
        pass

    def action(self, state):
        """Chọn hành động một cách xác định (dùng khi đánh giá)."""
        pass

    def value(self, state, action):
        """Tính giá trị của cặp (trạng thái, hành động)."""
        pass

    # ----------------------------------------------------------------------
    # Đánh giá mô hình
    # ----------------------------------------------------------------------

    def evaluate(self, env, eval_episodes=10, max_steps_per_episode=10):
        """
        Đánh giá tác tử trên nhiều tập trong môi trường.

        Trả về:
            - rewards_all: Danh sách phần thưởng trung bình mỗi tập.
            - infos_all: Thông tin bổ sung cho từng bước.
        """
        rewards_all = []
        infos_all = []

        for _ in range(eval_episodes):
            episode_rewards = []
            episode_infos = []
            state = env.reset(predict=False)

            for _ in range(max_steps_per_episode):
                action = self.action(state)
                state, reward, done, _, info = env.step(action)
                done = done[0] if isinstance(done, list) else done

                episode_rewards.append(reward)
                episode_infos.append(info)

                if done:
                    break
                state = env.reset(predict=False)

            self.log(f"Phần thưởng mỗi bước: {episode_rewards}")

            if episode_rewards:
                avg_reward = np.mean(episode_rewards)
                self.log(f"Trung bình phần thưởng: {avg_reward}")
                rewards_all.append(avg_reward)

            infos_all.append(episode_infos)

        return rewards_all, infos_all

