import os
import numpy as np
import torch
import threading

from core.task_generator import TaskGenerator

class DDQNTrainer:
    """
    Lớp quản lý và điều phối quá trình huấn luyện các tác tử (agent) DDQN trong môi trường mô phỏng.

    Lưu trữ tiến trình huấn luyện, quản lý cập nhật mạng mục tiêu, ghi lại hiệu suất,
    và tự động lưu mô hình theo từng giai đoạn.
    """

    def __init__(
        self,
        env,
        agents,
        score_window_size,
        max_episode_length,
        update_frequency,
        save_dir,
        use_thread=True,
        use_detach_thread=True,
        train_start_factor=1
    ):
        """
        Khởi tạo đối tượng DDQNTrainer với các tham số huấn luyện và môi trường.

        Args:
            env: Môi trường mô phỏng nơi các agent tương tác.
            agents (list): Danh sách các agent DDQN được huấn luyện.
            score_window_size (int): Kích thước cửa sổ tính điểm trung bình để đánh giá hiệu suất.
            max_episode_length (int): Số bước (timesteps) tối đa trong một episode.
            update_frequency (int): Tần suất cập nhật mạng mục tiêu của DDQN.
            save_dir (str): Đường dẫn thư mục để lưu checkpoint và kết quả huấn luyện.
            use_thread (bool): Cho phép huấn luyện đa luồng (mặc định: True).
            use_detach_thread (bool): Cho phép tách luồng huấn luyện chạy nền (mặc định: True).
            train_start_factor (int): Hệ số nhân cho giai đoạn khởi động trước khi bắt đầu huấn luyện.
        """
        # --- Cấu hình môi trường & agent ---
        self.__env = env
        self.__agents = agents

        # --- Tham số huấn luyện ---
        self.__score_window_size = score_window_size
        self.__max_episode_length = max_episode_length
        self.__update_frequency = update_frequency
        self.__start_train_step = train_start_factor * update_frequency

        # --- Cấu hình lưu trữ ---
        self.__save_dir = save_dir
        self.__checkpoints_dir = os.path.join(self.__save_dir, "checkpoints")

        # --- Ghi log quá trình huấn luyện ---
        self.__score_history = []
        self.__task_history = []
        self.__benefit_history = []
        self.__episode_length_history = []

        # --- Bộ đếm và trạng thái ---
        self.__time_step = 0
        self.__episode_count = 0
        self.__best_score = 0
        self.__eta = 0.0  # có thể dùng làm hệ số điều chỉnh học

        # --- Đa luồng ---
        self.__use_thread = use_thread
        self.__use_detach_thread = use_detach_thread

        # --- Bộ sinh nhiệm vụ ---
        self.__task_generator = TaskGenerator(15, env.get_map_obj())

        # --- Tạo thư mục checkpoint ---
        if not os.path.exists(self.__checkpoints_dir):
            os.makedirs(self.__checkpoints_dir)
        else:
            print(f"⚠️ Thư mục '{self.__checkpoints_dir}' đã tồn tại — các checkpoint mới sẽ được ghi đè.")

    def get_score_history(self):
        """Lấy lịch sử điểm số (reward) qua các episode"""
        return self.__score_history
    
    def set_time_step(self, time_step):
        """Thay đổi giá trị time_step"""
        self.__time_step = time_step

    def set_episode_count(self, episode_count):
        """Thay đổi giá trị episode count"""
        self.__episode_count = episode_count

    def step_env(self, actions, states):
        """
        Thực hiện hành động trong môi trường và trả về trạng thái, phần thưởng, cờ kết thúc và thông tin môi trường.
        """
        # Gọi môi trường thực thi hành động -> trả về thông tin sau bước
        env_info = self.__env.step(actions_by_vehicle = actions,states = states,  agents = self.__agents)

        # Tách các giá trị trả về từ môi trường
        next_states = env_info[0]   # Trạng thái kế tiếp
        rewards = env_info[1]       # Phần thưởng tương ứng với mỗi agent
        dones = env_info[2]         # Cờ kết thúc tập (episode)
        truncateds = env_info[3]    # Cờ cho biết bị dừng sớm
        env_data = env_info[4]      # Thông tin môi trường chi tiết
        action_dict = env_info[5]   # Hành động được ánh xạ cho từng agent

        # Trả về các thông tin cần thiết cho bước huấn luyện
        return next_states, rewards, dones, truncateds, env_data, action_dict
    
    def calculate_reward_optimized(self, reward):
        """
        Tính giá trị R bằng căn bậc hai của tổng bình phương hiệu giữa các phần tử trong danh sách reward.
        """
        sum_squared_diffs = 0  # Khởi tạo tổng bình phương hiệu
        for i in range(len(reward)):
            for j in range(i + 1, len(reward)):
                sum_squared_diffs += (reward[i] - reward[j]) ** 2  # Cộng dồn bình phương hiệu giữa r[i] và r[j]
        return np.sqrt(sum_squared_diffs)  # Trả về căn bậc hai của tổng

    def run_episode(self):
        """
        Thực thi một episode huấn luyện của hệ thống đa tác tử (multi-agent RL).

        Mô tả:
            - Mỗi agent sẽ quan sát trạng thái môi trường, chọn hành động.
            - Các hành động được thực hiện trong môi trường, nhận về phần thưởng, trạng thái kế tiếp.
            - Dữ liệu kinh nghiệm được lưu lại cho quá trình huấn luyện.
            - Cập nhật mạng (train model) theo tần suất định sẵn.

        Returns:
            scores (dict): Danh sách điểm thưởng (reward) cho từng tác tử theo thời gian.
        """

        # === Khởi tạo danh sách điểm thưởng cho từng phương tiện ===
        total_rewards = []
        self.__env.data = self.__env.get_environment_data()
        
        for _ in range(self.__env.data['num_vehicles']):
            total_rewards.append([])

        # === Khởi tạo lại môi trường và lấy trạng thái ban đầu ===
        env_info = self.__env.reset()
        current_states = env_info[0]

        # === Biến lưu các hành động được chọn để tránh trùng lặp ===
        used_actions = []

        # === Bắt đầu vòng lặp từng bước thời gian trong episode ===
        for step in range(self.__max_episode_length):
            print(step, "/", self.__max_episode_length)
            self.__time_step += 1

            # Các danh sách tạm để lưu thông tin hành động và trạng thái
            processed_states, chosen_actions, saved_actions, log_probs = [], [], [], []

            # --- Mỗi agent chọn hành động dựa trên trạng thái hiện tại ---
            for agent_idx, state_key in enumerate(current_states):
                agent = self.__agents[agent_idx]
                observation = np.reshape(current_states[state_key], (1, -1))
                torch_state = torch.from_numpy(observation).float()

                # Lấy hành động và xác suất log
                action, log_prob = agent.get_actions(torch_state, agent_idx)

                # Tránh chọn trùng hành động trong cùng timestep
                if any(torch.equal(torch_state, s) for s in processed_states) \
                    and int(np.argmax(action[1])) in used_actions:
                    continue

                used_actions.append(int(np.argmax(action[1])))
                processed_states.append(torch_state)
                log_probs.append(log_prob)
                chosen_actions.append(action)
                saved_actions.append(action[1])

            # --- Thực thi các hành động trong môi trường ---
            next_states, rewards, dones, truncated, _, action_dict = self.step_env(chosen_actions, current_states)

            # Đặt lại giá trị done cho từng tác tử
            dones = [dones] * len(current_states)

            # --- Lưu kinh nghiệm vào bộ nhớ học tập của từng tác tử ---
            for agent, state, action, log_prob, reward, done, next_state in zip(
                    self.__agents, processed_states, action_dict, log_probs,
                    rewards, dones, next_states):

                action_key = action_dict[action]
                flattened_state = state.view(-1)
                next_state_value = next_states[next_state]
                agent.add_memory(flattened_state, action_key, rewards[reward], next_state_value)

            # Cập nhật trạng thái mới
            current_states = next_states

            # --- Nếu đủ điều kiện thì huấn luyện agent ---
            if (self.__time_step > self.__agents[0].get_train_start() and
                self.__time_step > self.__start_train_step and
                self.__time_step % self.__env.data['max_missions_per_vehicle'] == 0):

                threads = []
                for idx, agent in enumerate(self.__agents):
                    if not self.__thread:
                        agent.train_model()
                    else:
                        update_thread = threading.Thread(target=agent.train_model)
                        if self.__detach_thread:
                            # Chạy thread nền (daemon) để huấn luyện song song
                            update_thread.daemon = True
                            update_thread.start()
                        else:
                            update_thread.start()
                            threads.append(update_thread)

                # Nếu không chạy detach thread, đợi tất cả thread kết thúc
                if not self.__detach_thread:
                    for idx, thread in enumerate(threads):
                        thread.join()

            # --- Cập nhật mạng mục tiêu định kỳ ---
            if self.__time_step > 0 and self.__time_step % 1000 == 0:
                for agent in self.__agents:
                    agent.update_target_model()

            # --- Ghi nhận phần thưởng ---
            for idx, reward in rewards.items():
                total_rewards[idx] += reward

            # --- Kết thúc episode nếu có agent hoàn thành ---
            if np.any(dones):
                break
        self.__score_history.append(total_rewards)
        return total_rewards