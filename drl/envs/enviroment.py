import json
import threading
import copy
import numpy as np
import gymnasium as gym
import time
from math import sqrt
from gymnasium.spaces import Box
from ray.tune.registry import register_env
from config.config import SEED_GLOBAL, mission_config
from core.its.mission import Mission
from core.its.vehicle import Vehicle
from core.task_generator import TaskGenerator
from drl.envs.data_loader import DataLoader
from drl.envs.utils import get_ideal_expected_reward


SEED = SEED_GLOBAL


class Environment(gym.Env):
    """
    Môi trường ITS (Intelligent Transportation System) đa tác nhân.

    Biến:
        data (dict): Dữ liệu đầu vào chứa thông tin về nhiệm vụ, phương tiện, bản đồ, đồ thị,...
        verbose (bool): Nếu True, in thông tin chi tiết quá trình xử lý.
        generator (np.random.Generator): Bộ sinh số ngẫu nhiên cho môi trường.
        __map_obj (Map): Bản đồ của môi trường ITS.
        vehicles (list[Vehicle]): Danh sách phương tiện trong môi trường.
        missions (list[Mission]): Danh sách nhiệm vụ trong môi trường.
        current_step (int): Bước hiện tại trong tập huấn luyện.
        max_steps (int): Số bước tối đa cho mỗi tập.
        _agent_ids (set): Tập các ID phương tiện (agent).
        action_space (Box): Không gian hành động, mỗi phương tiện chọn nhiệm vụ.
        observation_space (Box): Không gian quan sát dạng vector.
        action_memory (np.array): Theo dõi các nhiệm vụ đã được chọn.
        solution (list): Theo dõi nhiệm vụ đã được phân cho phương tiện nào.
        max_selection_turn (list): Số lượt tối đa còn lại để phương tiện chọn nhiệm vụ.
        done (bool): Trạng thái hoàn tất tập.
        tg (TaskGenerator): Bộ sinh nhiệm vụ ngẫu nhiên.
    """

    def __init__(self, data, verbose=True, map_obj=None, task_generator=None, max_steps=100):
        super().__init__()
        self.__data = data
        self.__verbose = verbose
        self.__max_steps = max_steps
        self.__current_step = 0
        self.__done = True

        # Bộ sinh số ngẫu nhiên
        self.__random = np.random.default_rng(SEED)
        self.__map_obj = map_obj

        # Khởi tạo danh sách phương tiện và nhiệm vụ
        self.__vehicles = self.init_vehicles()
        self.__missions = self.init_missions()

        # Theo dõi ID của các agent (phương tiện)
        self.__agent_ids = {f"vehicle_{i}" for i in range(self.__data["num_vehicles"])}

        # Không gian hành động & quan sát
        self.__action_space = Box(
            -np.inf, np.inf, shape=(data["total_missions"],), dtype="float32"
        )
        self.__observation_space = Box(-np.inf, np.inf, shape=(14604, 1), dtype="float32")

        # Theo dõi trạng thái nội bộ
        self.__action_memory = np.zeros(data["total_missions"], dtype=int)
        self.__solution = ["None"] * (
            mission_config["num_vehicles"] * mission_config["max_missions_per_vehicle"]
        )
        self.__max_selection_turn = [
            self.__data["max_missions_per_vehicle"]
        ] * self.__data["num_vehicles"]
        self.__task_generator = task_generator if task_generator is not None else TaskGenerator(15, self.__map_obj)

        # Giá trị phần thưởng trung bình lý tưởng cho mỗi phương tiện
        self.__ideal_avg_reward = get_ideal_expected_reward()

    def get_map_obj(self):
        """Trả về Bản đồ của môi trường ITS."""
        return self.__map_obj

    def get_observation_space(self):
        """Trả về Không gian quan sát dạng vector."""
        return self.__observation_space
    
    def get_action_space(self):
        """Trả về Không gian hành động, mỗi phương tiện chọn nhiệm vụ."""
        return self.__action_space

    def init_vehicles(self):
        """
        Khởi tạo các phương tiện (Vehicle) ngẫu nhiên trên các đoạn đường của bản đồ.

        Returns:
            list[Vehicle]: Danh sách phương tiện.
        """
        vehicles = []
        for i in range(self.__data["n_vehicles"]):
            segment = self.__random.choice(self.__data["segments"])
            vehicle = Vehicle(
                cpu_freq=0.5,
                current_position=segment.get_endpoints()[0],
                road_map=self.__map_obj,
                verbose=self.__verbose,
                tau=self.__data["tau"],
            )
            vehicles.append(vehicle)
        # Khởi tạo lại trạng thái của phương tiện cuối cùng
        vehicles[-1].reset()
        return vehicles

    def init_missions(self):
        """
        Khởi tạo danh sách các nhiệm vụ (missions) sử dụng đa luồng để tăng hiệu quả.

        Mỗi nhiệm vụ được tạo từ dữ liệu đầu vào `decoded_data`,
        thiết lập phụ thuộc (`depends`) và đăng ký các phương tiện (vehicles)
        để theo dõi nhiệm vụ này.

        Returns:
            list: Danh sách các Nhiệm vụ đã được khởi tạo.
        """
        mission_list = []  # Danh sách lưu trữ các nhiệm vụ
        lock = (
            threading.Lock()
        )  # Khóa để tránh tranh chấp khi thêm nhiệm vụ vào danh sách

        def create_mission(item):
            """
            Tạo một nhiệm vụ từ dữ liệu đầu vào và thêm vào danh sách missions.
            """
            mission = Mission(
                start_point=item["start_point"],
                end_point=item["end_point"],
                time_slot=1,
                graph=self.__data["graph"],
                verbose=self.__verbose,
            )
            mission.set_depend_mission(item["depends"])
            mission.register_observer(self.__vehicles)
            with lock:
                mission_list.append(mission)

        threads = []  # Danh sách luồng
        for item in self.__data["decoded_data"]:
            thread = threading.Thread(
                target=create_mission, args=(item,)
            )  # Tạo luồng cho mỗi nhiệm vụ
            threads.append(thread)
            thread.start()  # Bắt đầu luồng
        for thread in threads:
            thread.join()  # Chờ tất cả các luồng hoàn thành

        Mission.mission_counter = 0
        return mission_list
    
    def init_vehicles(self):
        """
        Khởi tạo danh sách phương tiện trong môi trường.
        
        Mỗi phương tiện sẽ được gán ngẫu nhiên một đoạn đường (segment)
        làm vị trí khởi đầu, đồng thời thiết lập các thông số cơ bản như
        tốc độ, bản đồ, và hệ số trễ (tau).
        
        Returns:
            List[Vehicle]: Danh sách đối tượng phương tiện đã khởi tạo.
        """
        vehicles = []
        for vehicle_index in range(self.__data["num_vehicles"]):
            # Chọn ngẫu nhiên một đoạn đường làm vị trí bắt đầu cho phương tiện
            start_segment = self.__random.choice(self.__data["segments"])
            
            # Tạo đối tượng phương tiện
            vehicle = Vehicle(
                cpu_freq=0.5,
                current_position=start_segment.get_endpoints()[0],
                road_map=self.__map_obj,
                verbose=self.__verbose,
                tau=self.__data["tau"],
            )
            
            vehicles.append(vehicle)

        # Đặt lại trạng thái phương tiện
        Vehicle.reset_vehicle_id_counter()

        return vehicles

    def reset(self, reload_config=True, for_prediction=False):
        """
        Đặt lại môi trường về trạng thái ban đầu.

        Thao tác này sẽ:
        - Tải lại dữ liệu nhiệm vụ nếu cần.
        - Khởi tạo lại các phương tiện (vehicles) và nhiệm vụ (missions).
        - Reset bộ nhớ hành động và giải pháp hiện tại.
        - Tạo lại không gian quan sát và thông tin cho từng agent.

        Args:
            reload_config (bool): Nếu True, tải lại cấu hình từ task_generator khi reset.
            for_prediction (bool): Nếu True, reset để dùng cho dự đoán, bỏ qua điều kiện done.

        Returns:
            observations (dict): Dictionary chứa quan sát cho mỗi phương tiện.
            infos (dict): Dictionary rỗng chứa thông tin phụ cho từng agent.
        """

        start = time.perf_counter()  # Bắt đầu đo thời gian reset

        if (self.__done and reload_config and not eval) or for_prediction:
            if self.__verbose:
                print("---------> reset", self.__done)
            self.__data = DataLoader.generate_config_not_from_file(
                self.__task_generator
            )
            self.__current_step = 0

        self.__vehicles = self.init_vehicles()
        self.__missions = copy.deepcopy(
            self.__data.get("missions", self.init_missions())
        )
        self.__agent_ids = {f"vehicle_{i}" for i in range(self.__data["num_vehicles"])}
        self.__action_memory = np.zeros(self.__data["total_missions"], dtype=int)
        self.__solution = ["None"] * (
            mission_config["num_vehicles"] * mission_config["max_missions_per_vehicle"]
        )
        self.max_selection_turn = [
            self.__data["max_missions_per_vehicle"]
        ] * self.__data["num_vehicles"]

        observations = self.get_observations()
        infos = {agent_id: {} for agent_id in self.__agent_ids}

        if self.__verbose:
            end = time.perf_counter()
            print("Reset time: {:.4f}s".format(end - start))

        return observations, infos

    def get_observations(self):
        """
        Tạo và trả về tập quan sát (observations) cho tất cả các phương tiện trong môi trường.

        Mỗi phương tiện (vehicle) sẽ nhận được một vector quan sát (obs_vec)
        bao gồm thông tin về:
            - Các đoạn đường (segments) và trạng thái của chúng
            - Độ dài của các nhiệm vụ (missions)
            - Phụ thuộc giữa các nhiệm vụ
            - Vị trí của các phương tiện khác
            - Bộ nhớ hành động (action_memory)
            - Thông tin phụ thuộc nhiệm vụ mà mỗi phương tiện đã chấp nhận

        Returns:
            dict: Từ điển chứa quan sát của từng phương tiện,
                có khóa là "vehicle_i" và giá trị là vector numpy 1 chiều.
        """
        observations = {}

        # Mỗi phần tử gồm: [độ dài đoạn đường, trạng thái hiện tại]
        segment_info = np.array(
            [
                [segment.get_distance(), segment.get_status()]
                for segment in self.__data["segments"]
            ],
            dtype=np.float32,
        )
        mission_lengths = np.array(
            [[mission.get_distance()[0]] for mission in self.__missions],
            dtype=np.float32,
        )

        # Lấy danh sách các phụ thuộc của từng nhiệm vụ
        num_missions_depends = np.zeros((self.__data["total_missions"], 10))
        for idx, mission in enumerate(self.__missions):
            depends = mission.get_dependencies()
            num_missions_depends[idx][: len(depends)] = depends

        # Thu gọn thông tin phụ thuộc nhiệm vụ cho từng phương tiện
        vehicle_depends = np.zeros(
            (self.__data["num_vehicles"], self.__data["max_missions_per_vehicle"], 10)
        )
        for i, vehicle in enumerate(self.__vehicles):
            for j, mission in enumerate(vehicle.get_accepted_missions()):
                vehicle_depends[i][j][
                    : len(mission.get_dependencies())
                ] = mission.get_dependencies()
        vehicle_depends = vehicle_depends.flatten()

        # Duyệt qua từng phương tiện để tạo vector quan sát
        for i, vehicle in enumerate(self.__vehicles):
            vehicle_positions = []

            # Lấy vị trí của tất cả các phương tiện khác (không bao gồm chính nó)
            for other_vehicle in self.__vehicles:
                if other_vehicle != vehicle:
                    x, y = other_vehicle.get_position().get_point()
                    vehicle_positions.extend([x, y])

            # Bổ sung thêm số 0 để vector vị trí có độ dài cố định
            vehicle_positions.extend(
                [0] * (len(self.__missions) * 2 - len(vehicle_positions))
            )
            vehicle_positions = np.array(vehicle_positions, dtype=np.float32)

            # Ghép các phần dữ liệu lại thành một vector quan sát hoàn chỉnh
            obs_vec = np.concatenate(
                [
                    segment_info.flatten() / 5000,
                    mission_lengths.flatten() / (5000 * sqrt(2)),
                    vehicle_positions / 5000,
                    num_missions_depends.flatten(),
                    self.__action_memory.flatten(),
                    vehicle_depends,
                ]
            )

            # Gán vector quan sát cho phương tiện tương ứng
            observations[f"vehicle_{i}"] = obs_vec

        # Trả về kết quả
        return observations

    def step(self, actions_by_vehicle, agents=None):
        """
        Thực hiện một bước (step) trong môi trường mô phỏng.

        Args:
            actions_by_vehicle (dict): Từ điển chứa hành động của từng phương tiện.
                Mỗi phần tử có dạng {index: (id, action_tensor)}.
            agents (list, optional): Danh sách các tác nhân (agents),
                có thể dùng cho cơ chế khám phá epsilon-greedy (nếu có).

        Returns:
            tuple:
                observations (dict): Quan sát mới của các phương tiện.
                rewards (dict): Phần thưởng cho từng phương tiện.
                done (bool): Cho biết tập (episode) đã kết thúc hay chưa.
                truncated (bool): Cờ dừng sớm (mặc định False).
                infos (dict): Thông tin bổ sung.
                executed_actions (dict): Hành động thực tế đã được thực hiện (sau khi kiểm tra hợp lệ).
        """
        rewards = {}
        executed_actions = {}

        # Áp dụng hành động cho từng phương tiện
        for vehicle_index, vehicle_id in enumerate(actions_by_vehicle):
            # Xác định agent
            agent = agents[vehicle_index] 

            # Lấy hành động thô từ mạng neural (tensor)
            raw_action_tensor = actions_by_vehicle[vehicle_index][1]

            # Chuyển tensor sang numpy và chọn hành động có giá trị cao nhất
            selected_action = int(np.argmax(raw_action_tensor.detach().numpy()))

            # Nếu nhiệm vụ này đã được chọn trước đó
            if self.__action_memory[selected_action] == 1:
                # Áp dụng phạt nhẹ để tránh chọn trùng
                rewards[vehicle_index] = [-0.01 * self.__ideal_avg_reward]
                executed_actions[vehicle_index] = selected_action

                # Lưu kinh nghiệm vào bộ nhớ học (replay buffer) của agent.
                agent.add_local_experience()
                agent.add_global_experience()

                continue

            if self.__action_memory[selected_action] == False:
                # Gán nhiệm vụ cho phương tiện tương ứng
                self.__vehicles[vehicle_index].assign_missions(selected_action, self.__missions)

                # Đánh dấu nhiệm vụ đã được chọn
                self.__action_memory[selected_action] = 1
                self.__solution[selected_action] = vehicle_index

                # Giảm lượt chọn còn lại cho phương tiện
                self.__max_selection_turn[vehicle_index] -= 1

            # Lưu hành động đã thực hiện
            executed_actions[vehicle_index] = selected_action

        # Reset tổng phần thưởng cho tất cả các xe, tính toán lại cho tập hợp danh sách nhiệm vụ mới
        for vehicle in self.__vehicles:
            vehicle.reset_total_reward()

        # Xử lý các nhiệm vụ cho đến khi tất cả hoàn thành ---
        all_missions_done = []
        is_process_done = False
        while not is_process_done:
            is_process_done = True
            for vehicle in self.__vehicles:
                mission_status = vehicle.process_mission(self.__missions)
                all_missions_done.append(mission_status)
                if mission_status is None:  # Nếu vẫn còn nhiệm vụ đang thực hiện
                    is_process_done = False

        # Tính phần thưởng cho từng phương tiện ---
        total_reward = 0
        for vehicle_index, vehicle in enumerate(self.__vehicles):
            vehicle_reward = vehicle.get_vehicle_profit()
            total_reward += vehicle_reward
            rewards[vehicle_index] += [vehicle_reward]

        # --- 4️⃣ Kiểm tra điều kiện kết thúc tập ---
        is_done = (
            self.__current_step >= self.__max_steps  # đạt bước tối đa
            or (self.__action_memory == 1).all()   # hoặc tất cả nhiệm vụ đã được chọn
        )
        self.__done = is_done
        self.__current_step += 1

        # Lấy quan sát mới sau khi thực hiện hành động ---
        observations = self.get_observations()

        # --- 6️⃣ Trả về kết quả ---
        return observations, rewards, is_done, False, all_missions_done, executed_actions
    
