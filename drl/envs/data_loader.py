import json

from core.geometry.graph import Graph
from core.geometry.point import Point
from core.its.task import Task
from core.map.map import Map

from config.config import map_config, mission_config, other_config


class DataLoader:
    """
    Class để load dữ liệu cho mô phỏng ITS:
    - Bản đồ (Map)
    - Nhiệm vụ (Mission)
    - Task offloading trên từng segment
    Hỗ trợ load từ file JSON hoặc khởi tạo mission không dùng file.
    """

    # Class attribute: Map cố định dùng chung
    shared_map = Map(
        total_roads=map_config["num_roads"],
        current_traffic_state=map_config["traffic_level"],
        from_file=map_config["from_file"],
    )

    def __init__(
        self,
        mission_file="./data/mission_information.json",
        task_file="./data/task_information.json",
        graph=None,
    ):
        """
        Khởi tạo loader, đọc map, tasks, missions từ file.

        Args:
            mission_file (str): Tên file JSON chứa thông tin mission
            graph (Graph, optional): Graph đã có, nếu None sẽ khởi tạo từ map segments
        """
        print("Load map and missions from file...")

        # Sử dụng map chia sẻ
        self.map = DataLoader.shared_map
        self.map.draw_segments()
        self.map.draw_map()
        self.map.draw_segments()

        self.segments = self.map.get_segments()

        # Khởi tạo graph nếu không được cung cấp
        self.graph = graph if graph is not None else Graph(self.segments)

        # Load offloading tasks từ file task_information.json
        with open(task_file, "r") as file:
            json_task_data = json.load(file, object_hook=self._decode_task)
        self._assign_tasks_to_segments(json_task_data)

        # Load mission từ file
        self.read_successful = True
        try:
            with open(mission_file, "r") as file:
                json_mission_data = json.load(file, object_hook=self._decode_mission)
                self.missions_data = json_mission_data
        except Exception as e:
            self.read_successful = False
            raise ValueError(f"Cannot read mission file: {e}")

    @staticmethod
    def _decode_task(obj):
        """Chuyển dữ liệu JSON task thành Task objects."""
        if "tasks" in obj:
            obj["tasks"] = [Task(*v) for k, v in obj["tasks"].items()]
        return obj

    @staticmethod
    def _decode_mission(obj):
        """Chuyển dữ liệu JSON mission thành Point objects."""
        if "start_point" in obj:
            obj["start_point"] = Point(obj["start_point"][0], obj["start_point"][1])
        else:
            raise ValueError("Wrong file: Mission is missing start_point")
        if "end_point" in obj:
            obj["end_point"] = Point(obj["end_point"][0], obj["end_point"][1])
        else:
            raise ValueError("Wrong file: Mission is missing end_point")
        return obj

    def _assign_tasks_to_segments(self, tasks_data):
        """Gán danh sách task cho từng segment trên map."""
        for item in tasks_data:
            segment_id = item["segment_id"]
            idx = self.segments.index(segment_id)
            segment = self.segments[idx]
            segment.set_offloading_tasks(item["tasks"])

    def get_graph_and_map(self):
        """Trả về graph và map."""
        return self.graph, self.map

    def get_mission_data(self):
        """Trả về mission data, graph, và map."""
        return self.missions_data, self.graph, self.map

    def generate_config_from_file(self):
        """
        Tạo config chuẩn hóa từ file đã load.

        Returns:
            dict: config sử dụng trong môi trường mô phỏng
            Graph: graph của map
            Map: bản đồ
        """
        if not self.read_successful:
            return False

        config = {
            "n_missions": mission_config["total_missions"],
            "n_vehicles": mission_config["num_vehicles"],
            "n_miss_per_vec": mission_config["max_missions_per_vehicle"],
            "decoded_data": self.missions_data,
            "segments": self.map.get_segments(),
            "graph": self.graph,
            "thread": other_config["apply_thread"],
            "detach_thread": other_config["apply_detach"],
            "score_window_size": other_config["score_window_size"],
            "tau": other_config["tau"],
        }
        print("Finish loading data from file.")
        return config, self.graph, self.map

    def generate_config_not_from_file(mission_generator):
        """
        Tạo config khi khởi tạo missions không từ file.

        Args:
            mission_generator: Object có phương thức gen_mission_non_file

        Returns:
            dict: config sử dụng trong môi trường mô phỏng
        """
        missions, graph, map_obj = mission_generator.generate_missions_no_file(
            mission_config["total_missions"]
        )
        config = {
            "total_missions": mission_config["total_missions"],
            "num_vehicles": mission_config["num_vehicles"],
            "max_missions_per_vehicle": mission_config["max_missions_per_vehicle"],
            "decoded_data": "",
            "segments": map_obj.get_segments(),
            "graph": graph,
            "missions": missions,
            "apply_thread": other_config["apply_thread"],
            "apply_detach": other_config["apply_detach"],
            "score_window_size": other_config["score_window_size"],
            "tau": other_config["tau"],
            "map": map_obj,
        }
        return config
