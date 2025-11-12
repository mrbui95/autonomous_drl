import os
import numpy as np
import torch
import threading
import matplotlib.pyplot as plt
import pandas as pd
import logging

from core.task_generator import TaskGenerator
from config.config import mission_config
from config.drl_config import ddqn_config

logger = logging.getLogger(__name__)


class DDQNTrainer:
    def __init__(
        self,
        env,
        agents,
        score_window_size,
        max_episode_length,
        update_frequency,
        save_dir,
        use_thread=True,
        detach_thread=True,
        train_start_factor=1,
    ):
        """
        Khởi tạo DDQNTrainer cho môi trường Multi-Agent.

        Args:
            env: Môi trường Multi-Agent.
            agents: Danh sách agent DDQN.
            score_window_size: Cửa sổ để tính reward trung bình.
            max_episode_length: Số step tối đa mỗi episode.
            update_frequency: Số timestep giữa các lần cập nhật target model.
            save_dir: Thư mục lưu checkpoint.
            use_thread: Sử dụng thread để huấn luyện agent không (default=True).
            detach_thread: Nếu dùng thread, thread có detach không (default=True).
            train_start_factor: Hệ số để bắt đầu train so với update_frequency.
        """

        # Môi trường và agent
        self.env = env
        self.agents = agents

        # Thông số huấn luyện
        self.score_window_size = score_window_size
        self.max_episode_length = max_episode_length
        self.update_frequency = update_frequency
        self.start_train_step = train_start_factor * update_frequency

        # Thư mục lưu checkpoint
        self.save_dir = save_dir
        self.checkpoints_dir = os.path.join(self.save_dir, "checkpoints")
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        else:
            logger.info(f"Folder '{self.checkpoints_dir}' already exists.")

        # Lịch sử huấn luyện
        self.score_history = []
        self.task_history = []
        self.benefit_history = []
        self.episode_length_history = []

        # Bộ đếm
        self.current_step = 0
        self.current_episode = 0

        # Thread control
        self.use_thread = use_thread
        self.detach_thread = detach_thread

        # Reward tracking
        self.max_reward = 0
        self.eta = 0  # dùng để tính reward tối ưu hóa

        # Task generator cho môi trường
        self.task_generator = TaskGenerator(15, env.map_obj)

    # step_env
    def execute_environment_step(self, actions, current_states):
        """
        Thực hiện các action trong môi trường và trả về thông tin trạng thái, reward và termination.

        Args:
            actions (list): Danh sách action được chọn bởi các agent.
            current_states (dict): Trạng thái hiện tại của từng agent.

        Returns:
            next_states (dict): Trạng thái mới sau khi thực hiện actions.
            rewards (dict): Reward nhận được cho từng agent.
            done_flags (bool): True nếu episode kết thúc.
            truncated_flags (bool): True nếu episode bị truncated.
            done_process_info (Any): Thông tin xử lý nhiệm vụ của từng vehicle.
            action_mapping (dict): Mapping các action được thực hiện.
        """
        logger.debug("=== Bắt đầu execute_environment_step ===")
        logger.debug(f"Số lượng hành động nhận được: {len(actions)}")

        # In chi tiết từng hành động của agent (debug)
        for i, action in enumerate(actions):
            try:
                logger.debug(f" -> Agent {i}: action = {action}")
            except Exception as e:
                logger.warning(f"Không thể in action của agent {i}: {e}")

        # In thông tin trạng thái hiện tại
        logger.debug(f"Số lượng trạng thái hiện tại: {len(current_states)}")
        logger.debug(f"Khóa trạng thái (5 đầu tiên): {list(current_states.keys())[:5]} ...")

        # Thực hiện action trong môi trường và nhận thông tin trả về
        env_step_info = self.env.step_env(actions, self.agents, current_states)
        logger.debug("Đã gọi self.env.step_env() thành công.")

        # Trích xuất thông tin quan trọng từ môi trường
        next_states = env_step_info[0]
        rewards = env_step_info[1]
        done_flags = env_step_info[2]
        truncated_flags = env_step_info[3]
        done_process_info = env_step_info[4]
        action_mapping = env_step_info[5]

        # --- Ghi log chi tiết kết quả ---
        logger.debug(f"Trạng thái tiếp theo: {len(next_states)} entries")
        logger.debug(f"Reward: {rewards}")
        logger.debug(f"Done: {done_flags}, Truncated: {truncated_flags}")

        return (
            next_states,
            rewards,
            done_flags,
            truncated_flags,
            done_process_info,
            action_mapping,
        )

    # step_env_ma
    def execute_multi_action_step(self, actions):
        """
        Thực hiện các action trong môi trường đa-action (multi-action) và
        trả về trạng thái, reward và thông tin episode.

        Args:
            actions (list): Danh sách action cho từng agent.

        Returns:
            next_states (dict): Trạng thái mới của từng agent.
            rewards (list): Reward nhận được cho từng agent.
            done_flags (bool): True nếu episode kết thúc.
            truncated_flags (bool): True nếu episode bị truncated.
            env_info (Any): Thông tin môi trường bổ sung.
        """

        # Thực hiện multi-action trong môi trường
        env_step_info = self.env.step_multi_agent(actions)

        # Trích xuất thông tin quan trọng từ môi trường
        next_states = env_step_info[0]
        rewards = env_step_info[1]
        done_flags = env_step_info[2]
        truncated_flags = env_step_info[3]

        return next_states, rewards, done_flags, truncated_flags, env_step_info

    # calculate_r_optimized
    def calculate_reward_dispersion(self, rewards):
        """
        Tính độ lệch tổng quát (dispersion) của reward giữa các agent.

        Args:
            rewards (list or array): Danh sách reward từ các agent.

        Returns:
            float: Căn bậc hai của tổng bình phương các hiệu reward (một dạng đo độ phân tán).
        """
        sum_squared_diffs = 0
        n = len(rewards)

        # Tính tổng bình phương hiệu của từng cặp reward
        for i in range(n):
            for j in range(i + 1, n):
                sum_squared_diffs += (rewards[i] - rewards[j]) ** 2

        return np.sqrt(sum_squared_diffs)

    # run_episode
    def run_single_episode(self):
        """
        Chạy một episode đầy đủ trong môi trường.

        Returns:
            scores (list): Danh sách reward theo từng agent trong episode.
        """
        n_agents = self.env.env_data["num_vehicles"]

        # Khởi tạo scores cho từng agent
        scores = [[] for _ in range(n_agents)]

        # Reset môi trường và lấy trạng thái ban đầu
        env_info = self.env.reset_environment()
        current_states = env_info[0]

        # Lưu các action đã thực hiện để tránh trùng
        executed_actions = []

        for t in range(self.max_episode_length):
            logger.info(f"current_step {t}/{self.max_episode_length}")
            self.current_step += 1

            # Danh sách trạng thái đã xử lý, action, log_probs
            processed_states, actions, log_probs = [], [], []

            # Lấy action cho từng agent
            for idx, state_key in enumerate(current_states):
                agent = self.agents[idx]
                state_array = np.reshape(current_states[state_key], (1, -1))
                state_tensor = torch.from_numpy(state_array).float()

                action, log_prob = agent.select_actions(state_tensor, idx)

                # Nếu trạng thái + action đã xuất hiện thì bỏ qua
                if (
                    any(torch.equal(state_tensor, s) for s in processed_states)
                    and int(np.argmax(action[1])) in executed_actions
                ):
                    continue

                executed_actions.append(int(np.argmax(action[1])))
                processed_states.append(state_tensor)
                log_probs.append(log_prob)
                actions.append(action)

            # Thực hiện action trong môi trường và nhận phản hồi
            next_states, rewards, done_flags, truncated, _, actions_dict = (
                self.execute_environment_step(actions, current_states)
            )
            done_flags = [done_flags] * len(current_states)

            # Cập nhật memory cho từng agent
            for (
                agent,
                state_tensor,
                action_tensor,
                log_prob,
                reward,
                done,
                next_state_key,
            ) in zip(
                self.agents,
                processed_states,
                actions,
                log_probs,
                rewards,
                done_flags,
                next_states,
            ):

                action_val = actions_dict[action_tensor]
                state_tensor = state_tensor.view(-1)
                next_state_tensor = next_states[next_state_key]
                agent.add_experience(
                    state_tensor, action_val, rewards[reward], next_state_tensor
                )

            current_states = next_states

            # Huấn luyện agent nếu đủ điều kiện
            if (
                self.agents[0].train_start < self.current_step
                and self.current_step > self.train_start
                and self.current_step % self.env.env_data["max_missions_per_vehicle"] == 0
            ):

                threads = []
                for agent in self.agents:
                    if not self.thread:
                        agent.train_model()
                    else:
                        thread = threading.Thread(target=agent.train_model)
                        if self.detach_thread:
                            thread.daemon = True
                            thread.start()
                        else:
                            thread.start()
                            threads.append(thread)
                if not self.detach_thread:
                    for thread in threads:
                        thread.join()

            # Cập nhật target model định kỳ
            if self.current_step % 1000 == 0:
                for agent in self.agents:
                    agent.update_target_model()

            # Lưu reward vào scores
            for idx, reward in rewards.items():
                scores[idx] += reward

            # Kết thúc episode nếu bất kỳ agent nào done
            if np.any(done_flags):
                break

        return scores

    # do_modify_reward
    def apply_modified_rewards(self, modify_data):
        """
        Cập nhật reward của các agent dựa trên thông tin modify_reward và
        thêm trải nghiệm vào memory.

        Args:
            modify_data (dict): Bao gồm state, action, current_wards, next_state,
                                modified_infor, dones theo từng current_step.
        Returns:
            int: 0 khi hoàn tất.
        """
        # Xác định current_step nào có thay đổi reward
        reward_changed_flags = [0] * len(modify_data["state"])
        completed_count = 0

        # Đếm tổng số mission đã hoàn thành
        for step_info in modify_data["modified_infor"]:
            for entry in step_info:
                if entry is not None:
                    completed_count += 1

        # Tính reward bổ sung dựa trên mission hoàn thành và các yếu tố
        for step_idx, step_info in enumerate(modify_data["modified_infor"]):
            print('step_idx', step_idx, ', step_info', step_info)
            for entry in step_info:
                if entry is None:
                    continue
                vehicle_id, mission_id, n_remove_depends, n_waiting, profit = entry

                for agent_idx, actions in enumerate(modify_data["action"]):
                    # Kiểm tra nếu action agent khớp mission
                    if (
                        actions[vehicle_id] == mission_id
                        and modify_data["current_wards"][agent_idx][vehicle_id][0] >= 0
                    ):
                        # Tính reward bổ sung
                        additional_reward = (
                            (vehicle_id + 1) / mission_config["num_vehicles"]
                        ) * (
                            (mission_config["max_missions_per_vehicle"] - agent_idx)
                            * n_remove_depends
                            * 50
                            - n_waiting * 50
                        ) + completed_count * mission_config[
                            "n_mission"
                        ]
                        print(
                            "Reward update -> Vehicle:",
                            vehicle_id,
                            "Completed:",
                            completed_count,
                            "Removed dependencies:",
                            n_remove_depends,
                            "Waiting:",
                            n_waiting,
                            "Old reward:",
                            modify_data["current_wards"][agent_idx][vehicle_id],
                            "Added reward:",
                            additional_reward,
                        )

                        modify_data["current_wards"][agent_idx][vehicle_id][0] = (
                            profit + additional_reward
                        )
                        break

                reward_changed_flags[step_idx] = True

        # Cập nhật memory cho từng agent
        for step_idx, state_dict in enumerate(modify_data["state"]):
            for agent_idx, vehicle_key in enumerate(state_dict):
                action_val = modify_data["action"][step_idx][agent_idx]
                current_reward = modify_data["current_wards"][step_idx][agent_idx]
                next_state = modify_data["next_state"][step_idx][vehicle_key]
                done_flag = modify_data["dones"][step_idx][agent_idx]

                if reward_changed_flags[step_idx] and action_val != -1:
                    # Thêm vào cả global memory và local memory
                    self.agents[agent_idx].add_global_experience(
                        state_dict[vehicle_key],
                        action_val,
                        current_reward,
                        next_state,
                        done_flag,
                    )
                    self.agents[agent_idx].add_experience(
                        state_dict[vehicle_key],
                        action_val,
                        current_reward,
                        next_state,
                        done_flag,
                    )
                elif action_val != -1:
                    # Nếu không thay đổi reward, vẫn thêm -100
                    self.agents[agent_idx].add_experience(
                        state_dict[vehicle_key],
                        action_val,
                        [-100],
                        next_state,
                        done_flag,
                    )
                    self.agents[agent_idx].add_global_experience(
                        state_dict[vehicle_key],
                        action_val,
                        [-100],
                        next_state,
                        done_flag,
                    )

        return 0

    # run_episode_modify_reward
    def run_episode_with_reward_adjustment(self):
        """
        Chạy một episode duy nhất với việc điều chỉnh phần thưởng cho bài toán học tăng cường đa tác tử (multi-agent DRL).

        Trả về:
            scores (list): Danh sách phần thưởng của từng agent theo từng bước thời gian.
        """
        # Khởi tạo danh sách điểm thưởng cho mỗi agent
        scores = [[] for _ in range(self.env.env_data['num_vehicles'])]
        logger.debug(f"[INIT] Số lượng agent: {len(scores)}")  

        # Reset môi trường và lấy trạng thái ban đầu
        env_info = self.env.reset_environment()
        states = env_info[0]
        logger.debug("[ENV] Môi trường đã reset, trạng thái ban đầu nhận được.")

        # Lưu thông tin để điều chỉnh phần thưởng sau
        modify_data = {
            "step": [], "state": [], "action": [], "current_wards": [],
            "next_state": [], "modified_infor": [], "dones": []
        }

        # Lịch sử hành động để tránh trùng lặp
        action_history = []

        for t in range(self.max_episode_length):
            logger.info(f"Current_step {t}/{self.max_episode_length}")
            self.current_step += 1

            processed_states, actions, actions_save, log_probs = [], [], [], []

            # --- Lấy hành động cho mỗi agent ---
            for agent_idx, state_key in enumerate(states):
                agent = self.agents[agent_idx]
                obs = torch.from_numpy(np.reshape(states[state_key], (1, -1))).float()
                action, log_prob = agent.select_actions(obs, agent_idx)

                # Avoid duplicate states and actions
                if any(torch.equal(obs, ps) for ps in processed_states) \
                    and int(np.argmax(action[1])) in action_history:
                    logger.debug(f"[WARN] Bỏ qua agent {agent_idx} vì trùng hành động/trạng thái.")
                    continue

                processed_states.append(obs)
                log_probs.append(log_prob)
                actions.append(action)
                actions_save.append(action[1])
                action_history.append(int(np.argmax(action[1])))

            logger.debug(f"[INFO] Tổng số hành động hợp lệ ở bước {t}: {len(actions)}")

            # Thực hiện hành động trong môi trường
            next_states, rewards, dones, truncated, modified_info, actions = self.execute_environment_step(actions, states)
            dones = [dones] * len(states)
            logger.debug(f"[ENV] Môi trường trả về rewards: {rewards}")

            # Lưu dữ liệu cho giai đoạn điều chỉnh phần thưởng
            modify_data['step'].append(t)
            modify_data['state'].append(states)
            modify_data['action'].append(actions)
            modify_data['current_wards'].append(rewards)
            modify_data['next_state'].append(next_states)
            modify_data['modified_infor'].append(modified_info)
            modify_data['dones'].append(dones)

            # Huấn luyện các agent nếu đạt tần suất cập nhật
            if self.agents[0].train_start < self.current_step > self.start_train_step \
                and self.current_step % self.env.env_data['max_missions_per_vehicle'] == 0:
                logger.debug("[TRAIN] Bắt đầu huấn luyện agent...")
                threads = []
                for idx, agent in enumerate(self.agents):
                    if not self.thread:
                        agent.train_model()
                    else:
                        t_thread = threading.Thread(target=agent.train_model)
                        if self.detach_thread:
                            t_thread.daemon = True
                            t_thread.start()
                        else:
                            t_thread.start()
                            threads.append(t_thread)
                if not self.detach_thread:
                    for thread in threads:
                        thread.join()
                logger.debug("[TRAIN] Hoàn thành huấn luyện cho batch hiện tại.")

            # Cập nhật mạng mục tiêu định kỳ
            if self.current_step > 0 and self.current_step % 1000 == 0:
                logger.debug("[SYNC] Cập nhật target network cho các agent.")
                for agent in self.agents:
                    agent.update_target_model()

            # Cập nhật điểm thưởng thô
            for agent_idx, reward in rewards.items():
                scores[agent_idx] += reward
            logger.debug(f"[REWARD] Điểm thưởng cập nhật: {scores}")

            # Kiểm tra kết thúc episode
            if np.any(dones):
                logger.debug("[DONE] Một hoặc nhiều agent đã hoàn thành nhiệm vụ, kết thúc episode.")
                break
            
            # Cập nhật trạng thái
            states = next_states

        # Áp dụng điều chỉnh phần thưởng sau khi episode kết thúc
        logger.debug("[POST] Áp dụng điều chỉnh phần thưởng...")
        self.apply_modified_rewards(modify_data)
        logger.debug("[POST] Hoàn thành điều chỉnh phần thưởng.")

        logger.info("=== Kết thúc run_episode_with_reward_adjustment ===\n")
        return scores

    # run_episode_ma
    def run_multi_action_episode(self):
        """
        Runs a multi-action episode for all agents.
        Agents select missions; if multiple agents select the same mission,
        the later agent receives a penalty.
        
        Returns:
            rewards (list): Rewards for each agent at the end of the episode.
        """
        total_missions_per_agent = self.env.env_data['max_missions_per_vehicle']
        n_agents = self.env.env_data['num_vehicles']

        # Initialize reward storage and counters
        scores = [[] for _ in range(n_agents)]
        remaining_selections = np.array([total_missions_per_agent] * n_agents)
        action_order = [0] * n_agents
        agent_memory = {i: [[], [], [], [], []] for i in range(n_agents)}  # state, action, reward, next_state, order

        # Reset environment and get initial states
        env_info = self.env.reset_environment()
        states = env_info[0]

        # Track actions assigned
        assigned_actions = [0] * self.env.env_data['total_missions']
        first_queue_list = []
        iteration = 0
        max_free_select = 5

        while (self.env.action_memory == 0).any():
            for agent_idx, state_key in enumerate(states):
                agent = self.agents[agent_idx]
                obs = torch.from_numpy(np.reshape(states[state_key], (1, -1))).float()

                if remaining_selections[agent_idx] <= 0:
                    continue

                action, log_prob = agent.select_actions(obs, agent_idx)
                # epsilon-greedy selection
                if agent.epsilon > self.env.rng.random() or iteration > max_free_select:
                    action = self.env.rng.integers(0, agent.action_size)
                else:
                    action = int(np.argmax(action[1]))

                # Avoid duplicates in memory
                mem = agent_memory[agent_idx]
                if any(torch.equal(obs, s) for s in mem[0]) \
                    and any(torch.equal(obs, ns) for ns in mem[3]) \
                    and action in mem[1]:
                    continue

                mem[0].append(obs)  # state
                mem[3].append(obs)  # next_state placeholder
                mem[1].append(action)  # action

                # Penalty if mission already selected
                if self.env.action_memory[action]:
                    mem[2].append(-0.01 * np.mean(scores[agent_idx]) if scores[agent_idx] else -0.01)
                    mem[4].append(-1)
                    continue

                self.env.action_memory[action] = 1
                is_first_queue = len(self.env.missions[action].get_dependencies()) == 0
                if is_first_queue:
                    first_queue_list.append(action)

                next_state = self.env.get_multi_agent_observations(first_queue_list, move_vehicle_pos=is_first_queue)
                mem[2].append(0)
                mem[3][-1] = torch.from_numpy(np.reshape(next_state[state_key], (1, -1))).float()
                mem[4].append(action_order[agent_idx])
                assigned_actions[action] = (action_order[agent_idx], agent_idx)
                action_order[agent_idx] += 1
                remaining_selections[agent_idx] -= 1

            iteration += 1
            states = self.env.get_multi_agent_observations(first_queue_list)

        # Remove empty placeholders
        assigned_actions = [a for a in assigned_actions if a != 0]

        # Step environment for all actions
        _, rewards, dones, truncated, _ = self.execute_multi_action_step(assigned_actions)
        dones = [dones] * n_agents

        # Add experiences to agent memories
        for agent_idx, mem in agent_memory.items():
            reward = rewards[agent_idx]
            reduce_factor = reward / total_missions_per_agent
            if reward / (total_missions_per_agent * 100) > 1.0:
                reduce_factor = 0

            for i in range(len(mem[0])):
                if mem[2][i] == 0 and mem[4][i] != -1:
                    self.agents[agent_idx].add_experience(mem[0][i], mem[1][i],
                                                    reward + mem[2][i] - mem[4][i] * reduce_factor,
                                                    mem[3][i])
                else:
                    self.agents[agent_idx].add_experience(mem[0][i], mem[1][i], mem[2][i], mem[3][i])

        # Train agents if conditions met
        if self.agents[0].train_start < self.current_step > self.start_train_step:
            threads = []
            for idx, agent in enumerate(self.agents):
                if not self.thread:
                    agent.train_model()
                else:
                    t_thread = threading.Thread(target=agent.train_model)
                    if self.detach_thread:
                        t_thread.daemon = True
                        t_thread.start()
                    else:
                        t_thread.start()
                        threads.append(t_thread)
            if not self.detach_thread:
                for thread in threads:
                    thread.join()

        # Periodically update target networks
        if self.current_step > 0 and self.current_step % 5000 == 0:
            for agent in self.agents:
                agent.update_target_model()

        rewards = np.expand_dims(np.array(rewards), 1).tolist()
        self.current_step += 1
        return rewards

    # step
    def run_episode_step(self):
        """
        Thực hiện một tập (episode) huấn luyện trong môi trường.

        Tuỳ theo cấu hình, episode có thể chạy với reward đã được điều chỉnh
        hoặc chạy bình thường.

        Cập nhật:
            - self.score_history: danh sách tổng reward của từng agent mỗi episode
            - self.max_score: tổng reward cao nhất quan sát được
            - self.episode_length_history: độ dài của từng episode
        """
        # Tăng bộ đếm episode
        self.current_episode += 1

        # Chạy episode dựa trên thiết lập modify_reward
        if ddqn_config['modify_reward']:
            logger.info("Episode chạy có điều chỉnh reward")
            rewards_per_timestep = self.run_episode_with_reward_adjustment()
        else:
            logger.info("Episode chạy mà không điều chỉnh reward")
            rewards_per_timestep = self.run_single_episode()
            

        # Tính tổng reward của từng agent trong episode
        total_rewards_per_agent = np.sum(rewards_per_timestep, axis=1)

        # Lưu lịch sử reward và độ dài episode
        self.score_history.append(total_rewards_per_agent)
        self.max_score = max(total_rewards_per_agent)
        self.episode_length_history.append(len(rewards_per_timestep))

    # step_ma
    def run_multi_action_episode(self):
        """
        Thực hiện một tập (episode) huấn luyện cho môi trường đa hành động (multi-action).

        Mỗi agent sẽ chọn nhiệm vụ. Nếu nhiều agent chọn cùng một hành động,
        agent chọn sau sẽ bị trừ điểm (penalty).

        Cập nhật:
            - self.score_history: danh sách tổng reward của từng agent mỗi episode
            - self.episode_length_history: độ dài của từng episode
        """
        # Tăng bộ đếm episode
        self.current_episode += 1

        # Chạy episode multi-action
        rewards_per_timestep = self.run_multi_action_episode()

        # Tính tổng reward của từng agent trong episode
        total_rewards_per_agent = np.sum(rewards_per_timestep, axis=1)

        # Lưu lịch sử reward và độ dài episode
        self.score_history.append(total_rewards_per_agent)
        self.episode_length_history.append(len(rewards_per_timestep))

    # save
    def save_models(self):
        """
        Lưu mô hình actor-critic của tất cả agent.

        Mỗi agent sẽ được lưu thành file riêng trong thư mục checkpoint với
        tên file có định dạng: agent_{index}_{episode}.pth
        """
        for agent_idx, agent in enumerate(self.agents):
            # Tạo tên file checkpoint cho agent hiện tại
            checkpoint_path = f"{self.checkpoints_dir}/agent_{agent_idx}_{self.current_episode}.pth"
            
            # Lưu mô hình của agent
            agent.save_model(checkpoint_path)

    def print_status(self):
        """
        In thông tin phần thưởng trung bình và độ dài tập trung hiện tại của các episode.

        Hiển thị:
            - Trung bình phần thưởng của từng agent trong cửa sổ score_window_size.
            - Trung bình phần thưởng tối đa.
            - Tổng phần thưởng trung bình.
            - Độ dài trung bình của episode.
        """
        # Tính trung bình phần thưởng của từng agent trong cửa sổ gần nhất
        mean_reward_agent = np.mean(
            self.score_history[-self.score_window_size:],
            axis=0
        )

        # Tạo chuỗi thông tin phần thưởng từng agent
        agent_info_str = ''.join(
            f'Mean Reward Agent_{i}: {mean_reward_agent[i]:.2f}, '
            for i in range(len(self.agents))
        )

        # Tính trung bình phần thưởng tối đa trong cửa sổ
        mean_max_reward = np.max(
            self.score_history[-self.score_window_size:], axis=1
        ).mean()

        # Tính độ dài episode trung bình trong cửa sổ
        mean_episode_len = np.mean(
            self.episode_length_history[-self.score_window_size:]
        ).item()

        # In trạng thái hiện tại ra terminal
        print(
            f'\033[1mEpisode {self.current_episode} - '
            f'Mean Max Reward: {mean_max_reward:.2f}\033[0m'
            f'\n\t{agent_info_str}\n\t'
            f'Mean Total Reward: {mean_reward_agent.sum():.2f}, '
            f'Mean Episode Length: {mean_episode_len:.1f}'
        )

    # plot
    def df_scores(self):
        """
        Vẽ đồ thị học tập (learning curve) cho các agent:
            - Trung bình động phần thưởng của từng agent.
            - Trung bình động phần thưởng tối đa (Max Reward).

        Các cải tiến:
            - Tránh dùng màu nền giống background (ví dụ trắng).
            - Lưu đồ thị và dữ liệu phần thưởng ra thư mục lưu trữ.
        """
        # Khởi tạo DataFrame từ lịch sử phần thưởng
        columns = [f'Agent {i}' for i in range(len(self.agents))]
        df_scores = pd.DataFrame(self.score_history, columns=columns)
        df_scores['Max'] = df_scores.max(axis=1)  # Cột phần thưởng tối đa

        # Khởi tạo figure và trục
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.set_title('Learning Curve: Multi-Agent DDQN', fontsize=28)
        ax.set_xlabel('Episode', fontsize=21)
        ax.set_ylabel('Score', fontsize=21)

        # Vẽ trung bình động phần thưởng của từng agent (không bao gồm cột Max)
        df_scores.rolling(self.score_window_size).mean().iloc[:, :-1].plot(
            ax=ax,
            colormap='tab10',
            legend=True
        )

        # Vẽ trung bình động phần thưởng tối đa bằng màu đỏ
        df_scores['Max'].rolling(self.score_window_size).mean().plot(
            ax=ax,
            color='red',
            linewidth=2,
            label='Max Reward'
        )

        # Thêm lưới, chú thích và bố cục
        ax.grid(color='gray', linewidth=0.2)
        ax.legend(fontsize=13)
        plt.tight_layout()

        # Lưu đồ thị và dữ liệu phần thưởng
        fig.savefig(os.path.join(self.save_dir, 'scores.png'))
        df_scores.to_csv(os.path.join(self.save_dir, '_reward.csv'))

        # Đóng figure để giải phóng bộ nhớ
        plt.close()

    
