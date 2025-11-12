import json
import threading
import copy
import numpy as np
import gymnasium as gym
import time
import logging

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

logger = logging.getLogger(__name__)


class Environment(gym.Env):

    def __init__(
        self, env_data, verbose=True, map_obj=None, task_generator=None, max_steps=100
    ):
        """
        Khởi tạo môi trường Environment.

        Args:
            env_data (dict): Dữ liệu môi trường (vehicle, mission, segment,...)
            verbose (bool): Nếu True sẽ in log.
            map_obj: Bản đồ môi trường.
            task_generator (TaskGenerator hoặc None): Nếu None sẽ khởi tạo TaskGenerator mới.
            max_steps (int): Số bước tối đa trong một episode.
        """
        super().__init__()

        # Dữ liệu và cấu hình
        self.env_data = env_data
        self.verbose = verbose
        self.map_obj = map_obj
        self.current_step = 0
        self.max_steps = max_steps
        self.done = True

        # Task generator
        if task_generator is None:
            self.task_generator = TaskGenerator(15, self.map_obj)
        else:
            self.task_generator = task_generator

        # Random generator để sinh các giá trị ngẫu nhiên
        self.rng = np.random.default_rng(SEED)

        # Khởi tạo vehicles và missions
        self.vehicles = self.initialize_vehicles()
        self.missions = self.initialize_missions()

        # Agent IDs
        self.agent_ids = {f"vehicle_{i}" for i in range(self.env_data["num_vehicles"])}

        # Action và Observation spaces
        self.action_space = Box(
            -np.inf, np.inf, shape=(env_data["total_missions"],), dtype="float32"
        )
        self.observation_space = Box(
            -np.inf, np.inf, shape=(6816, 1), dtype="float32"
        )  # chiều có thể thay đổi tùy data

        # Bộ nhớ action và giải pháp
        self.action_memory = np.zeros(env_data["total_missions"], dtype=int)
        self.solution = ["None"] * (
            mission_config["num_vehicles"] * mission_config["max_missions_per_vehicle"]
        )
        self.max_selection_turn = [env_data["max_missions_per_vehicle"]] * env_data[
            "num_vehicles"
        ]

        # Giá trị phần thưởng trung bình lý tưởng cho mỗi phương tiện
        self.ideal_avg_reward = get_ideal_expected_reward()

    # init_missions
    def initialize_missions(self):
        """
        Khởi tạo danh sách các đối tượng Mission từ dữ liệu môi trường.
        Sử dụng threading để tạo mission song song, đảm bảo thread-safe khi thêm vào danh sách.
        """
        missions_list = []
        lock = threading.Lock()  # Khóa để truy cập danh sách missions thread-safe

        def create_single_mission(mission_data):
            """
            Tạo một Mission từ mission_data và thêm vào missions_list.
            """
            mission = Mission(
                mission_data["start_point"],
                mission_data["end_point"],
                time_slot=1,
                graph=self.env_data["graph"],
                verbose=self.verbose,
            )
            # Thiết lập phụ thuộc cho mission
            mission.set_depend_mission(mission_data["depends"])
            # Đăng ký các vehicle làm observer
            mission.register_observer(self.vehicles)

            # Thêm mission vào danh sách với khóa để tránh race condition
            with lock:
                missions_list.append(mission)

        # Tạo và khởi chạy thread cho từng mission
        threads = []
        for mission_data in self.env_data["decoded_data"]:
            t = threading.Thread(target=create_single_mission, args=(mission_data,))
            threads.append(t)
            t.start()

        # Chờ tất cả thread hoàn thành
        for t in threads:
            t.join()

        # Reset missionID chung cho tất cả mission
        Mission.mission_counter = 0

        return missions_list

    # init_vehicles
    def initialize_vehicles(self):
        """
        Khởi tạo danh sách các đối tượng Vehicle dựa trên dữ liệu môi trường.
        Chọn ngẫu nhiên segment khởi đầu cho mỗi vehicle và reset trạng thái cuối cùng.
        """
        vehicles_list = []

        for i in range(self.env_data["num_vehicles"]):
            # Chọn một segment ngẫu nhiên làm vị trí xuất phát
            start_segment = self.rng.choice(self.env_data["segments"])
            start_point = start_segment.get_endpoints()[0]

            # Tạo đối tượng Vehicle với thông số mặc định và thông tin môi trường
            vehicle = Vehicle(
                cpu_freq=0.5,
                current_position=start_point,
                road_map=self.map_obj,
                verbose=self.verbose,
                tau=self.env_data["tau"],
            )
            vehicles_list.append(vehicle)

        # Reset trạng thái vehicle cuối cùng (hoặc có thể dùng reset cho tất cả nếu cần)
        Vehicle.reset_vehicle_id_counter()

        return vehicles_list

    # reset
    def reset_environment(self, reload_file=True, predict=False):
        """
        Reset môi trường về trạng thái ban đầu.
        - Tải lại dữ liệu nhiệm vụ nếu cần.
        - Khởi tạo lại danh sách vehicles và missions.
        - Trả về observations và thông tin rỗng cho các agent.
        """
        start_time = time.perf_counter()

        # Nếu cần reset từ file hoặc chế độ predict
        if (self.done and reload_file and not eval) or predict:
            if self.verbose:
                print("---------> Reset môi trường, done =", self.done)
            # Sinh lại cấu hình nhiệm vụ
            data_loader = DataLoader(mission_file=self.missions)
            self.env_data = data_loader.generate_config_not_from_file(
                self.task_generator
            )
            self.current_step = 0

        file_reset_time = time.perf_counter()

        # Khởi tạo lại danh sách vehicles
        self.vehicles = self.initialize_vehicles()

        # Khởi tạo lại danh sách missions
        if "missions" in self.env_data:
            self.missions = copy.deepcopy(self.env_data["missions"])
        else:
            self.missions = self.initialize_missions()

        # Tạo tập ID các agent
        self.agent_ids = {f"vehicle_{i}" for i in range(self.env_data["num_vehicles"])}

        # Reset bộ nhớ hành động và solution
        self.action_memory = np.zeros(self.env_data["total_missions"], dtype=int)
        self.solution = ["None"] * (
            mission_config["num_vehicles"] * mission_config["max_missions_per_vehicle"]
        )

        # Lấy observation ban đầu
        observations = self.get_observations()
        infos = {agent_id: {} for agent_id in self.agent_ids}

        # Reset lượt chọn tối đa cho mỗi vehicle
        self.max_selection_turn = [
            self.env_data["max_missions_per_vehicle"]
        ] * self.env_data["num_vehicles"]

        end_time = time.perf_counter()
        if self.verbose:
            print(
                f"Thời gian sinh file: {file_reset_time - start_time:.4f}s, Reset object: {end_time - file_reset_time:.4f}s"
            )

        return observations, infos

    # reset_for_meta
    def reset_environment_meta(self, reload_file=True, predict=False):
        """
        Reset môi trường cho meta-learning.
        - Tải lại dữ liệu nhiệm vụ nếu reload_file=True hoặc predict=True.
        - Khởi tạo lại vehicles, missions và action memory.
        - Trả về observations, thông tin rỗng cho các agent và dữ liệu môi trường.
        """
        start_time = time.perf_counter()

        # Nếu cần reset từ file hoặc chế độ predict
        if reload_file or predict:
            if self.verbose:
                print("---------> Reset môi trường (meta), done =", self.done)
            # Sinh lại cấu hình nhiệm vụ
            data_loader = DataLoader(mission_file=self.missions)
            self.env_data = data_loader.generate_config_not_from_file(
                self.task_generator
            )
            self.current_step = 0

        file_reset_time = time.perf_counter()

        # Khởi tạo lại danh sách vehicles
        self.vehicles = self.initialize_vehicles()

        # Khởi tạo lại danh sách missions
        if "missions" in self.env_data:
            self.missions = copy.deepcopy(self.env_data["missions"])
        else:
            self.missions = self.initialize_missions()

        # Tạo tập ID các agent
        self.agent_ids = {f"vehicle_{i}" for i in range(self.env_data["num_vehicles"])}

        # Reset bộ nhớ hành động và solution
        self.action_memory = np.zeros(self.env_data["total_missions"], dtype=int)
        self.solution = ["None"] * (
            mission_config["n_vehicle"] * mission_config["max_missions_per_vehicle"]
        )

        # Lấy observation ban đầu
        observations = self.get_observations()
        infos = {agent_id: {} for agent_id in self.agent_ids}

        # Reset lượt chọn tối đa cho mỗi vehicle
        self.max_selection_turn = [
            self.env_data["max_missions_per_vehicle"]
        ] * self.env_data["num_vehicles"]

        end_time = time.perf_counter()
        if self.verbose:
            print(
                f"Thời gian sinh file: {file_reset_time - start_time:.4f}s, Reset object: {end_time - file_reset_time:.4f}s"
            )

        return observations, infos, self.env_data

    # get_observations
    def get_observations(self):
        """
        Tạo observation cho từng vehicle trong môi trường.
        Observation bao gồm:
        - Thông tin các segments (độ dài, trạng thái)
        - Độ dài các mission
        - Vị trí các vehicle khác (không bao gồm chính vehicle)
        - Quan hệ phụ thuộc giữa các mission
        - Bộ nhớ hành động (action_memory)
        - Quan hệ phụ thuộc giữa các mission đang được vehicle giữ
        Trả về dict: { "vehicle_i": observation_vector }
        """
        observations = {}

        # --- Chuẩn bị thông tin segment và mission một lần ngoài loop ---
        segment_info = np.array(
            [
                [segment.get_distance(), segment.get_status()]
                for segment in self.env_data["segments"]
            ],
            dtype=np.float32,
        )
        mission_lengths = np.array(
            [[mission.get_distance()[0]] for mission in self.missions], dtype=np.float32
        )

        # Quan hệ phụ thuộc giữa các mission, tối đa 10 depends
        mission_depends_array = np.zeros(
            (self.env_data["total_missions"], 10), dtype=np.float32
        )
        for idx, mission in enumerate(self.missions):
            depends = mission.get_dependencies()
            mission_depends_array[idx, : len(depends)] = depends

        max_vehicle_positions_size = (
            len(self.missions) * 2
        )  # kích thước tối đa vị trí các vehicle

        # Padding segment_info nếu số lượng mission > segment
        if len(self.missions) > segment_info.shape[0]:
            padding = np.zeros(
                (len(self.missions) - segment_info.shape[0], 2), dtype=np.float32
            )
            segment_info = np.vstack([segment_info, padding])

        # Quan hệ mission mà mỗi vehicle đang giữ
        vehicle_missions_depends_array = np.zeros(
            (
                self.env_data["num_vehicles"],
                self.env_data["max_missions_per_vehicle"],
                10,
            ),
            dtype=np.float32,
        )
        for v_idx, vehicle in enumerate(self.vehicles):
            accepted_missions = vehicle.get_accepted_missions()
            for m_idx, mission in enumerate(accepted_missions):
                vehicle_missions_depends_array[
                    v_idx, m_idx, : len(mission.get_dependencies())
                ] = mission.get_dependencies()
        vehicle_missions_depends_array = vehicle_missions_depends_array.flatten()

        # --- Tạo observation cho từng vehicle ---
        for v_idx, vehicle in enumerate(self.vehicles):
            # Vị trí các vehicle khác
            other_positions = []
            for other_vehicle in self.vehicles:
                if other_vehicle != vehicle:
                    x, y = other_vehicle.get_position().get_point()
                    other_positions.extend([x, y])
            # Padding nếu chưa đủ max size
            other_positions.extend(
                [0] * (max_vehicle_positions_size - len(other_positions))
            )
            other_positions = np.array(other_positions, dtype=np.float32)

            # Chuẩn bị các phần của observation
            action_memory = self.action_memory.flatten().astype(np.float32)
            segment_info_flat = segment_info.flatten() / 5000
            mission_lengths_flat = mission_lengths.flatten() / (5000 * np.sqrt(2))
            other_positions_flat = other_positions.flatten() / 5000
            mission_depends_flat = mission_depends_array.flatten()

            # Xác định độ dài các trường để ghi nhớ index
            self.idx_dict_obs = {
                "segment_info": 0,
                "mission_lengths": len(segment_info_flat),
                "vehicle_positions": len(segment_info_flat) + len(mission_lengths_flat),
                "num_missions_depends_array": len(segment_info_flat)
                + len(mission_lengths_flat)
                + len(other_positions_flat),
                "action_memory": len(segment_info_flat)
                + len(mission_lengths_flat)
                + len(other_positions_flat)
                + len(mission_depends_flat),
                "vehicle_missions_depends_array": len(segment_info_flat)
                + len(mission_lengths_flat)
                + len(other_positions_flat)
                + len(mission_depends_flat)
                + len(action_memory),
            }

            # Padding tất cả các phần để cùng độ dài
            max_len = max(
                len(segment_info_flat),
                len(mission_lengths_flat),
                len(other_positions_flat),
                len(mission_depends_flat),
                len(action_memory),
                len(vehicle_missions_depends_array),
            )
            segment_info_padded = np.pad(
                segment_info_flat, (0, max_len - len(segment_info_flat))
            )
            mission_lengths_padded = np.pad(
                mission_lengths_flat, (0, max_len - len(mission_lengths_flat))
            )
            vehicle_positions_padded = np.pad(
                other_positions_flat, (0, max_len - len(other_positions_flat))
            )
            mission_depends_padded = np.pad(
                mission_depends_flat, (0, max_len - len(mission_depends_flat))
            )
            action_memory_padded = np.pad(
                action_memory, (0, max_len - len(action_memory))
            )
            vehicle_missions_depends_padded = np.pad(
                vehicle_missions_depends_array,
                (0, max_len - len(vehicle_missions_depends_array)),
            )

            # Nối tất cả thành observation vector cho vehicle
            observations[f"vehicle_{v_idx}"] = np.concatenate(
                [
                    segment_info_padded,
                    mission_lengths_padded,
                    vehicle_positions_padded,
                    mission_depends_padded,
                    action_memory_padded,
                    vehicle_missions_depends_padded,
                ]
            )

        return observations

    # get_ma_observations
    def get_multi_agent_observations(self, depend_index=None, move_vehicle_pos=False):
        """
        Tạo observation cho từng vehicle, có thể loại bỏ một số phụ thuộc (depend_index)
        và/hoặc di chuyển vị trí vehicle tới mission mục tiêu (move_vehicle_pos=True).

        Args:
            depend_index (list[int], optional): Các mission index cần loại khỏi phụ thuộc. Default là None.
            move_vehicle_pos (bool): Nếu True, vị trí vehicle được đặt tới mission cuối trong depend_index.

        Returns:
            dict: { "vehicle_i": observation_vector }
        """
        if depend_index is None:
            depend_index = []

        observations = {}

        # --- Chuẩn bị thông tin segment và mission ---
        segment_info = np.array(
            [
                [segment.get_distance(), segment.get_status()]
                for segment in self.env_data["segments"]
            ],
            dtype=np.float32,
        )
        mission_lengths = np.array(
            [[mission.get_distance()[0]] for mission in self.missions], dtype=np.float32
        )

        # Quan hệ phụ thuộc giữa các mission, tối đa 10 depends
        mission_depends_array = np.zeros(
            (self.env_data["total_missions"], 10), dtype=np.float32
        )
        for idx, mission in enumerate(self.missions):
            depends = [d for d in mission.get_dependencies() if d not in depend_index]
            mission_depends_array[idx, : len(depends)] = depends

        max_vehicle_positions_size = len(self.missions) * 2

        # Padding segment_info nếu số lượng mission > số segment
        if len(self.missions) > segment_info.shape[0]:
            padding = np.zeros(
                (len(self.missions) - segment_info.shape[0], 2), dtype=np.float32
            )
            segment_info = np.vstack([segment_info, padding])

        # Quan hệ mission mà mỗi vehicle đang giữ
        vehicle_missions_depends_array = np.zeros(
            (self.env_data["num_vehicles"], self.env_data["max_missions_per_vehicle"], 10),
            dtype=np.float32,
        )
        for v_idx, vehicle in enumerate(self.vehicles):
            accepted_missions = vehicle.get_accepted_missions()
            # Loại bỏ các mission trong depend_index
            accepted_missions = [m for m in accepted_missions if m not in depend_index]
            for m_idx, mission in enumerate(accepted_missions):
                vehicle_missions_depends_array[
                    v_idx, m_idx, : len(mission.get_dependencies())
                ] = mission.get_dependencies()
        vehicle_missions_depends_array = (
            vehicle_missions_depends_array.flatten() / self.env_data["total_missions"]
        )

        # --- Tạo observation cho từng vehicle ---
        for v_idx, vehicle in enumerate(self.vehicles):
            # Vị trí các vehicle khác
            vehicle_positions = []
            for other_vehicle in self.vehicles:
                if other_vehicle != vehicle:
                    x, y = other_vehicle.get_position().get_point()
                    vehicle_positions.extend([x, y])

            # Vị trí vehicle hiện tại
            if move_vehicle_pos and depend_index:
                x, y = self.missions[depend_index[-1]].get_end_point().get_point()
            else:
                x, y = vehicle.get_position().get_point()
            vehicle_positions.extend([x, y])

            # Padding vị trí để đủ max size
            vehicle_positions.extend(
                [0, 0] * (max_vehicle_positions_size - len(vehicle_positions) // 2)
            )
            vehicle_positions = np.array(vehicle_positions, dtype=np.float32)

            # Chuẩn bị các phần khác của observation
            action_memory = np.array(self.action_memory, dtype=np.float32).flatten()
            segment_info_flat = segment_info.flatten() / 5000
            mission_lengths_flat = mission_lengths.flatten() / (5000 * np.sqrt(2))
            vehicle_positions_flat = vehicle_positions.flatten() / 5000
            mission_depends_flat = (
                mission_depends_array.flatten() / self.env_data["total_missions"]
            )

            # Định nghĩa index dict (cho các phần của observation)
            self.idx_dict_obs = {
                "segment_info": 0,
                "mission_lengths": len(segment_info_flat),
                "vehicle_positions": len(segment_info_flat) + len(mission_lengths_flat),
                "num_missions_depends_array": len(segment_info_flat)
                + len(mission_lengths_flat)
                + len(vehicle_positions_flat),
                "action_memory": len(segment_info_flat)
                + len(mission_lengths_flat)
                + len(vehicle_positions_flat)
                + len(mission_depends_flat),
                "vehicle_missions_depends_array": len(segment_info_flat)
                + len(mission_lengths_flat)
                + len(vehicle_positions_flat)
                + len(mission_depends_flat)
                + len(action_memory),
            }

            # Padding tất cả phần để cùng độ dài
            max_len = max(
                len(segment_info_flat),
                len(mission_lengths_flat),
                len(vehicle_positions_flat),
                len(mission_depends_flat),
                len(action_memory),
                len(vehicle_missions_depends_array),
            )
            segment_info_padded = np.pad(
                segment_info_flat, (0, max_len - len(segment_info_flat))
            )
            mission_lengths_padded = np.pad(
                mission_lengths_flat, (0, max_len - len(mission_lengths_flat))
            )
            vehicle_positions_padded = np.pad(
                vehicle_positions_flat, (0, max_len - len(vehicle_positions_flat))
            )
            mission_depends_padded = np.pad(
                mission_depends_flat, (0, max_len - len(mission_depends_flat))
            )
            action_memory_padded = np.pad(
                action_memory, (0, max_len - len(action_memory))
            )
            vehicle_missions_depends_padded = np.pad(
                vehicle_missions_depends_array,
                (0, max_len - len(vehicle_missions_depends_array)),
            )

            # Ghép tất cả thành observation vector cho vehicle
            observations[f"vehicle_{v_idx}"] = np.concatenate(
                [
                    segment_info_padded,
                    mission_lengths_padded,
                    vehicle_positions_padded,
                    mission_depends_padded,
                    action_memory_padded,
                    vehicle_missions_depends_padded,
                ]
            )

        return observations

    # get_single_observation
    def get_single_observation(self):
        """
        Tạo observation vector tổng hợp cho toàn bộ môi trường (single observation)
        bao gồm thông tin segment, mission, vị trí vehicle, action_memory và
        phụ thuộc của các mission mà vehicle đang giữ.

        Returns:
            np.ndarray: Observation vector 1D.
        """
        # --- Chuẩn bị thông tin segment và mission ---
        segment_info = np.array(
            [[segment.get_distance(), segment.get_status()] for segment in self.env_data["segments"]],
            dtype=np.float32,
        )
        mission_lengths = np.array(
            [[mission.get_distance()[0]] for mission in self.missions], dtype=np.float32
        )

        # Mảng phụ thuộc các mission, tối đa 10 depends
        num_missions_depends_array = np.zeros(
            (self.env_data["total_missions"], 10), dtype=np.float32
        )
        for idx, mission in enumerate(self.missions):
            depends = mission.get_dependencies()
            num_missions_depends_array[idx, : len(depends)] = depends

        # --- Chuẩn bị vị trí vehicle và mission mà vehicle đang giữ ---
        max_vehicle_positions_size = len(self.missions) * 2

        # Mảng phụ thuộc các mission mà vehicle đang giữ
        vehicle_missions_depends_array = np.zeros(
            (self.env_data["num_vehicles"], self.env_data["max_missions_per_vehicle"], 10),
            dtype=np.float32,
        )
        for v_idx, vehicle in enumerate(self.vehicles):
            accepted_missions = vehicle.get_accepted_missions()
            for m_idx, mission in enumerate(accepted_missions):
                vehicle_missions_depends_array[
                    v_idx, m_idx, : len(mission.get_dependencies())
                ] = mission.get_dependencies()
        vehicle_missions_depends_array = (
            vehicle_missions_depends_array.flatten() / 30
        )  # Chuẩn hóa

        # --- Chuẩn bị vị trí các vehicle (không tính self) ---
        vehicle_positions = []
        for other_vehicle in self.vehicles:
            x, y = other_vehicle.get_position().get_point()
            vehicle_positions.extend([x, y])
        # Padding để đủ kích thước max_vehicle_positions_size
        vehicle_positions.extend(
            [0] * (max_vehicle_positions_size - len(vehicle_positions))
        )
        vehicle_positions = np.array(vehicle_positions, dtype=np.float32).reshape(-1, 2)

        # --- Chuẩn hóa các thông tin ---
        segment_info_flat = segment_info.flatten() / 5000
        mission_lengths_flat = mission_lengths.flatten() / (5000 * np.sqrt(2))
        vehicle_positions_flat = vehicle_positions.flatten() / 5000
        num_missions_depends_flat = num_missions_depends_array.flatten() / 30
        action_memory_flat = np.array(self.action_memory, dtype=np.float32).flatten()

        # --- Ghép tất cả thành observation vector ---
        obs = np.concatenate(
            [
                segment_info_flat,
                mission_lengths_flat,
                vehicle_positions_flat,
                num_missions_depends_flat,
                action_memory_flat,
                vehicle_missions_depends_array,
            ]
        )

        return obs

    # update_action
    def update_action_memory(self, action_probs):
        """
        Cập nhật action_memory dựa trên xác suất hành động được chọn.
        """
        # Lấy chỉ số hành động có xác suất cao nhất
        selected_action = int(np.argmax(action_probs[1]))

        # Đánh dấu hành động này đã được thực hiện
        self.action_memory[selected_action] = 1

    # update_mem_obs
    def update_observation_with_actions(self, observations, vehicle_id):
        """
        Cập nhật action_memory vào quan sát của một vehicle.
        """
        # Chuyển action_memory hiện tại sang dạng mảng 1 chiều
        current_actions = np.array(self.action_memory).flatten()

        # Lấy vị trí bắt đầu trong vector quan sát của vehicle
        start_idx = self.idx_dict_obs["action_memory"]

        # Cập nhật action_memory vào đúng vị trí trong quan sát
        observations[f"vehicle_{vehicle_id}"][
            start_idx : start_idx + len(current_actions)
        ] = current_actions

        return observations

    # get_solution
    def check_solution_completion(self):
        """
        Kiểm tra tiến độ hoàn thành các nhiệm vụ và xác định xem tất cả nhiệm vụ đã kết thúc chưa.
        """
        total_completed_tasks = 0
        all_vehicles_done = True

        for vehicle in self.vehicles:
            # Cộng số nhiệm vụ mà vehicle đã hoàn thành
            total_completed_tasks += vehicle.get_total_completed()

            # Debug: in thời gian kiểm soát và trạng thái vehicle
            print(vehicle.is_on_time(), vehicle.get_control_time())

            # Nếu có bất kỳ vehicle nào chưa xong, tiếp tục vòng lặp
            if vehicle.is_on_time() is True:
                all_vehicles_done = False

        # In số nhiệm vụ hoàn thành và trạng thái solution hiện tại
        print("Số nhiệm vụ đã hoàn thành:", total_completed_tasks)
        print("Solution hiện tại:", self.solution)

        # Trả về True nếu tất cả nhiệm vụ đã hoàn tất, False nếu còn vehicle chưa xong
        return all_vehicles_done

    # step
    def step_env(self, actions_dict, agents=None, states=None):
        """
        Thực hiện một bước trong môi trường với các hành động của từng vehicle.

        Args:
            actions_dict (dict): Dictionary chứa hành động cho từng vehicle.
            agents (list, optional): Danh sách agent để xử lý exploration.
            states (dict, optional): Trạng thái dùng cho cập nhật bộ nhớ của agent.

        Returns:
            obs (dict): Quan sát mới sau bước.
            rewards (dict): Phần thưởng thu được cho từng vehicle.
            done (bool): True nếu tập episode kết thúc.
            truncated (bool): Thông tin truncation (giả sử False ở đây).
            done_process_info (list): Thông tin xử lý nhiệm vụ của từng vehicle.
            action_taken (dict): Hành động thực sự được thực hiện cho từng vehicle.
        """
        logger.debug("=== Bắt đầu step_env ===")
        logger.debug(f"Số lượng vehicle/action: {len(actions_dict)}")

        rewards = {}
        total_completed_tasks = 0
        total_system_profit = 0
        done_process_info_list = []
        action_taken = {}
        wrong_action_penalty = {idx: 0 for idx in range(len(self.vehicles))}

        # Xử lý hành động của từng vehicle
        for idx, v_id in enumerate(actions_dict):
            logger.debug(f"[ACTION] Vehicle {idx}, hành động raw: {actions_dict[idx]}")

            # Nếu tất cả nhiệm vụ đã được chọn
            if (self.action_memory == 1).all():
                logger.info("Tất cả nhiệm vụ đã được chọn. Đánh dấu action -1 cho vehicle còn lại.")
                for v in range(self.env_data["num_vehicles"]):
                    if v not in action_taken:
                        action_taken[v] = -1
                break

            # Kiểm tra lượt chọn còn lại
            elif self.max_selection_turn[idx] < 1:
                logger.debug(f"Vehicle {idx} đã hết lượt chọn, đánh dấu action -1.")
                action_taken[idx] = -1
                continue

            v_action = actions_dict[idx][1].detach().numpy()

            # Lựa chọn hành động dựa trên agent hoặc greedy
            if agents is not None:
                agent = agents[idx]
                if (agent.epsilon > self.rng.random()):
                    action = self.rng.integers(0, agent.action_dim)
                    logger.debug(f"Vehicle {idx} chọn action ngẫu nhiên: {action}")
                else:
                    action = int(np.argmax(v_action))
                    logger.debug(f"Vehicle {idx} chọn action greedy: {action}")
            else:
                logger.error("Agents không được cung cấp.")
                raise ValueError("Agents không được cung cấp.")

            # Nếu hành động đã được chọn trước đó
            if self.action_memory[action] == 1 and states is not None:
                agent.add_experience(
                    states[list(states.keys())[idx]],
                    action,
                    [-0.01],
                    states[list(states.keys())[idx]],
                    1,
                )
                agent.add_global_experience(
                    states[list(states.keys())[idx]],
                    action,
                    [-0.01],
                    states[list(states.keys())[idx]],
                    1,
                )
                wrong_action_penalty[idx] += -0.01
                action_taken[idx] = action
                logger.debug(f"Vehicle {idx} chọn action đã được chọn trước đó, phạt -0.01.")
                continue

            # Thực hiện hành động mới
            if self.action_memory[action] == 0:
                self.vehicles[idx].assign_missions(action, self.missions)
                self.action_memory[action] = 1
                self.solution[action] = idx
                self.max_selection_turn[idx] -= 1
                logger.debug(f"Vehicle {idx} thực hiện action {action}, cập nhật action_memory và solution.")

            action_taken[idx] = action

        # Reset reward của từng vehicle trước khi tính toán lại
        for vehicle in self.vehicles:
            vehicle.reset_total_reward()

        # Xử lý tiến trình nhiệm vụ
        while True:
            all_done = True
            for idx, vehicle in enumerate(self.vehicles):
                vehicle.process_mission(self.missions)
            for vehicle in self.vehicles:
                vehicle.check_and_move_ready_mission()
            for vehicle in self.vehicles:
                if vehicle.has_ready_missions():
                    all_done = False
            if all_done:
                break
        logger.debug("Hoàn tất xử lý tiến trình nhiệm vụ cho tất cả vehicle.")

        # Tính tổng phần thưởng và nhiệm vụ hoàn thành
        intime = False
        for idx, vehicle in enumerate(self.vehicles):
            prof_sys = vehicle.get_vehicle_profit()
            total_system_profit += prof_sys
            total_completed_tasks += vehicle.get_total_completed()
            if idx in rewards:
                rewards[idx].append(prof_sys)
            else:
                rewards[idx] = [prof_sys + wrong_action_penalty[idx]]
            if vehicle.is_on_time():
                intime = True
        logger.info(f"Tổng profit hệ thống: {total_system_profit}, nhiệm vụ hoàn thành: {total_completed_tasks}")


        # Kiểm tra điều kiện kết thúc episode
        done = (
            self.current_step >= self.max_steps
            or (np.array(self.action_memory) == 1).all()
            or not intime
        )
        truncated = False
        if done:
            self.done = True
        else:
            self.done = False

        self.current_step += 1
        obs = self.get_observations()

        if self.verbose:
            logger.debug(self.action_memory)
            logger.info(
                f"---> Tổng profit hệ thống {total_system_profit}, nhiệm vụ hoàn thành {total_completed_tasks}"
            )

        logger.debug("=== Kết thúc step_env ===\n")
        return obs, rewards, self.done, truncated, done_process_info_list, action_taken

    # step_ma
    def step_multi_agent(self, actions):
        """
        Thực hiện một bước trong môi trường đa agent (multi-agent).

        Args:
            actions (list or dict): Danh sách hành động hoặc tuple cho từng vehicle.

        Returns:
            obs (dict): Quan sát mới sau bước.
            rewards (list): Phần thưởng thu được cho từng vehicle.
            done (bool): True nếu tập episode kết thúc.
            truncated (bool): Thông tin truncation (giả sử False ở đây).
            infos (dict): Thông tin phụ trợ (trống trong ví dụ này).
        """
        total_completed_tasks = 0
        total_profit = 0
        rewards = [0] * self.env_data["num_vehicles"]

        # Gán nhiệm vụ cho tất cả vehicle
        for vehicle in self.vehicles:
            vehicle.assign_missions(actions, self.missions, mtuple=True)
        for vehicle in self.vehicles:
            vehicle.process_mission_orders()  # Sắp xếp thứ tự thực hiện nhiệm vụ

        # Xử lý các nhiệm vụ cho đến khi tất cả hoàn thành
        start_time = time.perf_counter()
        while True:
            terminate_loop = True
            for vehicle in self.vehicles:
                vehicle.process_mission(self.missions)
            for vehicle in self.vehicles:
                vehicle.check_and_move_ready_mission()
            for vehicle in self.vehicles:
                if vehicle.has_ready_missions():
                    terminate_loop = False
                    break
            if terminate_loop:
                break
        if self.verbose:
            print("Processing time: {:.4f} s".format(time.perf_counter() - start_time))

        # Tính phần thưởng và nhiệm vụ hoàn thành
        intime = False
        for idx, vehicle in enumerate(self.vehicles):
            total_completed_tasks += vehicle.get_total_completed()
            total_profit += vehicle.get_total_profit()
            rewards[idx] = vehicle.get_vehicle_profit()
            if vehicle.is_on_time():
                intime = True

        # Kiểm tra điều kiện kết thúc episode
        done = (
            self.current_step >= self.max_steps
            or not intime
            or sum(rewards) > self.ideal_avg_reward
        )
        truncated = False
        infos = {}  # Thông tin bổ sung (trống)

        self.done = done
        self.current_step += 1

        obs = self.get_observations()

        if self.verbose:
            print(self.action_memory)
            print(
                f"---> Rewards: {rewards}, Total Completed Tasks: {total_completed_tasks}, Total Profit: {total_profit}"
            )

        return obs, rewards, done, truncated, infos
