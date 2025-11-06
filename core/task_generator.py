import json
import time
import numpy as np

from core.geometry.graph import Graph
from core.its.mission import Mission
from core.its.task import Task
from utils.numpy_json_encoder import NumpyJSONEncoder

from config.config import map_config, task_config, mission_config


class TaskGenerator:
    """
    Lớp TaskGenerator dùng để sinh dữ liệu nhiệm vụ (Mission) và tác vụ (Task)
    cho mô phỏng hệ thống tính toán biên hoặc phương tiện tự hành.

    Các chức năng chính:
        - Sinh danh sách tác vụ (Task) theo từng đoạn đường (Segment)
        - Sinh danh sách nhiệm vụ (Mission) với ràng buộc phụ thuộc (dependency)
        - Cập nhật bản đồ (Map) và đồ thị (Graph)
        - Lưu thông tin ra file JSON để phục vụ huấn luyện hoặc mô phỏng

    Tham số:
        tau (float): Thời gian mô phỏng (đơn vị phút)
        road_map (Map): Đối tượng bản đồ chứa thông tin về các đoạn đường
        min_data_size (int): Kích thước dữ liệu nhỏ nhất của Task (KB)
        max_data_size (int): Kích thước dữ liệu lớn nhất của Task (KB)
        min_comp_size (int): Kích thước tính toán nhỏ nhất của Task (Mcycles)
        max_comp_size (int): Kích thước tính toán lớn nhất của Task (Mcycles)
    """

    def __init__(
        self,
        tau,
        road_map,
        min_data_size=100,  # KB
        max_data_size=500,  # KB
        min_comp_size=1,  # Mcycles
        max_comp_size=3,
    ):  # Mcycles
        self.__map = road_map
        self.__tau = tau
        self.__min_data_size = min_data_size
        self.__max_data_size = max_data_size
        self.__min_comp_size = min_comp_size
        self.__max_comp_size = max_comp_size

        # Vẽ bản đồ và các đoạn đường
        self.__map.draw_segments()
        self.__map.draw_map()

        # Khởi tạo bộ sinh số ngẫu nhiên cố định để tái lập mô phỏng
        self.__random = np.random.default_rng(seed=42)

        # Xây dựng đồ thị dựa trên các đoạn đường
        segments = self.__map.get_segments()
        self.graph = Graph(segments)

    # ----------------------------------------------------------------------
    def update_graph(self, segments):
        """Cập nhật lại đồ thị khi bản đồ thay đổi."""
        self.graph = Graph(segments)

    # ----------------------------------------------------------------------
    def generate_tasks(self, output_file="./data/task_information.json"):
        """
        Sinh danh sách các Task tương ứng với từng đoạn đường trong bản đồ
        và lưu vào file JSON.

        Cơ chế:
            - Tính thời gian chạy của mỗi đoạn dựa trên độ dài và tốc độ.
            - Sinh ngẫu nhiên số lượng Task trên mỗi đoạn tùy theo tốc độ và lưu lượng lambda.
            - Mỗi Task có kích thước dữ liệu và tính toán ngẫu nhiên trong phạm vi cấu hình.
        """
        segments = self.__map.get_segments()
        with open(output_file, "w") as f:
            f.write("[\n")

        for idx, seg in enumerate(segments):
            segment_info = {}
            status = seg.get_status()
            distance = seg.get_distance()
            task_rate, avg_speed = seg.get_info()

            # Tính tốc độ thực tế và thời gian chạy
            real_speed = avg_speed / ((status + 1) * 0.2)
            runtime = distance / real_speed
            num_tasks = int(task_rate * runtime)

            # Ghi thông tin đoạn đường
            segment_info["segment_id"] = seg.get_segment_id()
            segment_info["status"] = status
            segment_info["distance"] = distance
            segment_info["speed"] = real_speed
            segment_info["max_tasks"] = num_tasks

            # Sinh danh sách task cho đoạn đường
            tasks = {}
            for t_id in range(num_tasks):
                data_size = self.__random.integers(
                    self.__min_data_size, self.__max_data_size
                )
                comp_size = self.__random.integers(
                    self.__min_comp_size, self.__max_comp_size
                )
                task = Task(data_size=data_size, compute_size=comp_size)
                tasks[t_id] = task

            segment_info["tasks"] = tasks

            # Ghi vào file JSON
            json_obj = json.dumps(segment_info, indent=4, cls=NumpyJSONEncoder)
            with open(output_file, "a") as f:
                f.write(json_obj)
                if idx != len(segments) - 1:
                    f.write(",\n")

        with open(output_file, "a") as f:
            f.write("\n]")
        Task.id = 0

    def generate_missions_no_file(self, num_missions):
        """
        Sinh danh sách Mission trong bộ nhớ (không lưu file).

        Mỗi Mission có:
            - Điểm bắt đầu, điểm đích ngẫu nhiên
            - Phụ thuộc (dependencies) ngẫu nhiên giữa các Mission
            - Lợi nhuận (profit) ngẫu nhiên
        """
        start_time = time.perf_counter()

        if map_config["real_map"]:
            self.__map.update_real_map()

        vertices = list(set(self.graph.get_vertexes()))
        avoid_cycles = {}
        missions = []
        num_with_depends = 0

        for i in range(num_missions):
            start_p = self.__random.choice(vertices)
            end_p = self.__random.choice(vertices)
            distance = start_p.get_dis_to_point(end_p)
            travel_time = distance / task_config["max_speed"]

            # Đảm bảo nhiệm vụ hợp lệ theo giới hạn thời gian
            while (
                travel_time > task_config["tau"] * 60
                and travel_time < 50 / task_config["max_speed"]
            ):
                start_p = self.__random.choice(vertices)
                end_p = self.__random.choice(vertices)
                distance = start_p.get_dis_to_point(end_p)
                travel_time = distance / task_config["max_speed"]

            dependencies = []
            if self.__random.uniform() > 0.4:
                num_depends = self.__random.integers(1, 3)
                for _ in range(num_depends):
                    dep_id = self.__random.integers(0, num_missions)
                    while dep_id == i:
                        dep_id = self.__random.integers(0, num_missions)
                    if dep_id not in dependencies:
                        if dep_id in avoid_cycles and i not in avoid_cycles[dep_id]:
                            dependencies.append(dep_id)
                        else:
                            dependencies.append(dep_id)
                num_with_depends += 1

            avoid_cycles[i] = dependencies
            profit = self.__random.integers(
                mission_config["reward_range"][0], mission_config["reward_range"][1]
            )

            mission = Mission(start_p, end_p, tslot=0, graph=self.graph)
            mission.set_depend_mission(dependencies)
            mission.set_profit(profit)
            mission.set_mission_id(i)
            missions.append(mission)

        Mission.mission_counter = 0
        print(f"Thời gian sinh: {time.perf_counter() - start_time:.3f}s")
        print(
            f"→ Nhiệm vụ có phụ thuộc: {num_with_depends}/{mission_config['total_missions']}"
        )

        return missions, self.graph, self.__map

    # ----------------------------------------------------------------------
    def generate_missions(self, num_missions, file_name="mission_information.json"):
        """
        Sinh danh sách Mission và lưu vào file JSON.

        Bao gồm thông tin:
            - Điểm bắt đầu, điểm đích
            - Danh sách phụ thuộc
            - Lợi nhuận
        """
        start_time = time.perf_counter()

        if map_config["real_map"]:
            self.__map.update_real_map()

        vertices = list(set(self.graph.get_vertexes()))
        avoid_cycles = {}

        with open(f"./data/{file_name}", "w") as f:
            f.write("[\n")

        for i in range(num_missions):
            mission_info = {"id": i}
            start_p = self.__random.choice(vertices)
            end_p = self.__random.choice(vertices)
            distance = start_p.get_dis_to_point(end_p)
            travel_time = distance / task_config["max_speed"]

            while (
                travel_time > task_config["time_limit"] * 60
                and travel_time < 50 / task_config["max_speed"]
            ):
                start_p = self.__random.choice(vertices)
                end_p = self.__random.choice(vertices)
                distance = start_p.get_dis_to_point(end_p)
                travel_time = distance / task_config["max_speed"]

            dependencies = []
            if self.__random.uniform() > 0.4:
                num_depends = self.__random.integers(1, 3)
                for _ in range(num_depends):
                    dep_id = self.__random.integers(0, num_missions)
                    while dep_id == i:
                        dep_id = self.__random.integers(0, num_missions)
                    if dep_id not in dependencies:
                        if dep_id in avoid_cycles and i not in avoid_cycles[dep_id]:
                            dependencies.append(dep_id)
                        else:
                            dependencies.append(dep_id)
            avoid_cycles[i] = dependencies

            profit = self.__random.integers(
                mission_config["reward_range"][0], mission_config["reward_range"][1]
            )
            mission_info.update(
                {
                    "start_point": start_p,
                    "end_point": end_p,
                    "dependencies": dependencies,
                    "profit": profit,
                }
            )

            json_obj = json.dumps(mission_info, indent=4, cls=NumpyJSONEncoder)
            with open(f"./data/{file_name}", "a") as f:
                f.write(json_obj)
                if i != num_missions - 1:
                    f.write(",\n")

        with open(f"./data/{file_name}", "a") as f:
            f.write("\n]")
        print(f"Thời gian sinh file: {time.perf_counter() - start_time:.3f}s")

    # ----------------------------------------------------------------------
    def generate_all(self, mission_file="mission_information.json"):
        """
        Cập nhật bản đồ, sinh Task và Mission mới.
        Dùng khi cần làm mới toàn bộ dữ liệu mô phỏng.
        """
        self.__map.update_real_map()
        self.generate_tasks()
        self.generate_missions(mission_config["n_mission"], file_name=mission_file)
