import matplotlib.pyplot as plt
import numpy as np
import time
import osmnx as ox

from config.config import map_config

from core.geometry.graph import Graph
from core.geometry.line import Line
from core.geometry.point import Point
from core.geometry.segment import Segment
from core.map.osm_map import RealMap
from core.map.road import Road


class Map:
    """
    Đại diện cho bản đồ thành phố gồm các tuyến đường, giao lộ và đoạn đường,
    hỗ trợ cả bản đồ sinh tổng hợp (synthetic) và bản đồ thực (real map).

    Thuộc tính:
    traffic_probs (list): Ma trận xác suất cho các trạng thái giao thông khác nhau.
    traffic_states (list): Danh sách các trạng thái giao thông (mã số).
    total_roads (int): Tổng số đường trong bản đồ.
    vertical_roads (int): Số đường theo trục dọc.
    horizontal_roads (int): Số đường theo trục ngang.
    current_traffic_state (int): Trạng thái giao thông hiện tại của bản đồ.
    max_vertical (int): Giá trị tọa độ dọc lớn nhất (chiều cao bản đồ).
    max_horizontal (int): Giá trị tọa độ ngang lớn nhất (chiều rộng bản đồ).
    intersections (dict): Dictionary ánh xạ các điểm giao cắt tới danh sách đường.
    random (np.random.Generator): Bộ tạo số ngẫu nhiên để đảm bảo tính lặp lại.
    boundary_roads (list): Danh sách các đường giới hạn (biên).
    roads (list): Danh sách tất cả các đường trong bản đồ.
    segments (list): Danh sách các đoạn đường được chia từ các đường.
    draw_segments_dict (dict): Dictionary dùng để vẽ các đoạn đường theo trạng thái giao thông.
    real_map (RealMap): Đối tượng bản đồ thực (nếu áp dụng).
    road_info (Any): Thông tin về các đường khi sử dụng bản đồ thực.
    intersection_list (list): Danh sách các điểm giao cắt.
    real_map_ready (bool): Cờ cho biết bản đồ thực đã được cập nhật/sẵn sàng hay chưa.
    distance_dict (dict): Dictionary lưu khoảng cách ngắn nhất giữa các đầu đoạn đường.
    """

    traffic_probs = [
        ["1.0", "0.0", "0.0", "0.0", "0.0"],  # Không tắc
        ["0.7", "0.2", "0.1", "0.0", "0.0"],  # Lưu thông bình thường
        ["0.4", "0.3", "0.1", "0.1", "0.1"],  # Giờ cao điểm
        ["0.1", "0.1", "0.2", "0.3", "0.3"],  # Tắc nghẽn
        ["0.0", "0.1", "0.1", "0.3", "0.5"],  # Tắc nghiêm trọng
    ]
    traffic_states = [0, 1, 2, 3, 4]

    def __init__(
        self,
        total_roads,
        height=5000,
        width=5000,
        current_traffic_state=0,
        from_file=False,
    ):
        """
        Khởi tạo bản đồ.
        Args:
            total_roads (int): Tổng số tuyến đường.
            height (int): Chiều cao bản đồ (tọa độ tối đa theo trục Y).
            width (int): Chiều rộng bản đồ (tọa độ tối đa theo trục X).
            traffic_level (int): Mức độ tắc nghẽn chung của bản đồ.
            from_file (bool): Có tải cấu hình từ file hay không.
        """
        # Số lượng đường dọc và ngang
        self.__total_roads = total_roads
        self.__num_vertical_roads = total_roads // 2
        self.__num_horizontal_roads = total_roads - self.__num_vertical_roads

        # Thông số tắc nghẽn
        self.__current_traffic_state = current_traffic_state
        self.__map_height = height
        self.__map_width = width

        # Danh sách giao điểm
        self.__intersections = {}  # {Point: [Road]}
        self.__random = np.random.default_rng(48)

        # Biên bản đồ
        self.__edges = [
            Road(Line(point_1=Point(0, 0), k=np.tan(90 * np.pi / 180))),
            Road(
                Line(
                    point_1=Point(self.__map_height, self.__map_width),
                    k=np.tan(90 * np.pi / 180),
                )
            ),
            Road(Line(point_1=Point(0, self.__map_width), k=np.tan(0))),
            Road(Line(point_1=Point(0, 0), k=np.tan(0))),
        ]

        # Khởi tạo bản đồ
        if not from_file:
            self.make_map()
        elif not map_config["real_map"]:
            self.load_map()

        self.__real_map_ready = False
        self.build()

    def load_map(self):
        """
        Đọc thông tin bản đồ từ file và khởi tạo các đối tượng Road tương ứng.
        Mỗi dòng trong file xác định một đường với tọa độ, độ dốc và trạng thái giao thông.
        """
        filepath = "./core/map/map_info.txt"
        with open(filepath, "r") as file:
            lines = file.readlines()

        self.__roads = []
        for line in lines:
            """
            Line có dạng [Id tuyến đường],[Tọa độ x của gốc],[Tọa độ y của gốc],[Hệ số góc k của đường thẳng]
            Ví dụ: 7,2002,0,6.005970377361733
            """
            line = line.strip()
            # Chọn trạng thái giao thông ngẫu nhiên dựa trên trạng thái bản đồ hiện tại
            road_state = self.__random.choice(
                Map.traffic_states, p=Map.traffic_probs[self.__current_traffic_state]
            )
            # Tách thông số đường từ file
            params = line.split(",")
            # Tạo đối tượng Line từ điểm và độ dốc
            line_obj = Line(
                point_1=Point(float(params[1]), float(params[2])), k=float(params[3])
            )
            # Tạo đối tượng Road với trạng thái giao thông
            road_obj = Road(line_obj, road_state)
            # Cập nhật ID đường theo file
            road_obj.update_id(int(params[0]))
            self.__roads.append(road_obj)

    def make_map(self):
        """
        Tạo bản đồ tổng hợp với các đường ngang và dọc.
        Mỗi đường có độ dài và góc ngẫu nhiên, đồng thời gán trạng thái giao thông.
        """
        self.__roads = []

        # --- Tạo các đường ngang ---
        x_pos, y_pos = 0, 0
        base_horizontal_line = Line(
            point_1=Point(x_pos, y_pos), k=np.tan(85 * np.pi / 180)
        )
        self.__roads.append(Road(base_horizontal_line))

        for _ in range(self.__num_horizontal_roads - 1):
            angle_variation = self.__random.uniform(-5, 5)  # biến thiên góc
            distance_increment = self.__random.integers(
                low=int(self.__map_width * 0.7 / self.__total_roads),
                high=int(3 * self.__map_width / self.__total_roads),
            )
            x_pos += distance_increment
            new_horizontal_line = base_horizontal_line.get_line_from_angle(
                point=Point(x_pos, y_pos), angle_deg=angle_variation
            )
            traffic_state = self.__random.choice(
                Map.traffic_states,
                p=Map.traffic_probs[self.__current_traffic_state],
            )
            self.__roads.append(Road(new_horizontal_line, traffic_state))
            base_horizontal_line = new_horizontal_line  # cập nhật đường cơ sở

        # --- Tạo các đường dọc ---
        x_pos, y_pos = 0, 0
        base_vertical_line = Line(
            point_1=Point(x_pos, y_pos), k=np.tan(1 * np.pi / 180)
        )
        self.__roads.append(Road(base_vertical_line))

        for _ in range(self.__num_vertical_roads - 1):
            angle_variation = self.__random.uniform(-5, 5)
            distance_increment = self.__random.integers(
                low=int(self.__map_height * 0.7 / self.__total_roads),
                high=int(3 * self.__map_height / self.__total_roads),
            )
            y_pos += distance_increment
            new_vertical_line = base_vertical_line.get_line_from_angle(
                point=Point(x_pos, y_pos), angle_deg=angle_variation
            )
            traffic_state = self.__random.choice(
                Map.traffic_states,
                p=Map.traffic_probs[self.__current_traffic_state],
            )
            self.__roads.append(Road(new_vertical_line, traffic_state))
            base_vertical_line = new_vertical_line  # cập nhật đường cơ sở

    def build(self):
        """
        Xây dựng bản đồ:
        - Nếu là bản đồ tổng hợp (synthetic), xác định các giao lộ và phân đoạn đường.
        - Nếu là bản đồ thực (real map), tạo HMap và lấy các phân đoạn, thông tin đường và giao lộ.
        """
        if not map_config["real_map"]:
            # Khởi tạo dictionary để vẽ phân đoạn theo trạng thái giao thông
            self.__segments_by_state = {0: [], 1: [], 2: [], 3: [], 4: []}

            # Cập nhật các giao lộ cho bản đồ
            self.update_intersections()

            # Reset danh sách phân đoạn
            self.__segments = []
            Segment.segment_counter = 0

            # Tạo các phân đoạn cho từng đường
            for road in self.__roads:
                road.build_segments()  # Phân đoạn từng đường
                road_segments = road.get_segments()
                self.__segments += road_segments

                # Lưu trữ các điểm phân đoạn để vẽ
                for segment in road_segments:
                    status = segment.get_status()
                    self.__segments_by_state[status].append(segment.get_points())

        else:
            # Bản đồ thực: tạo RealMap
            self.__real_map = RealMap(
                center_point=map_config["real_center_point"], radius=map_config["radius"]
            )

            # Xây dựng bản đồ thực
            segments, road_info_updated, intersections = self.__real_map.build_map(
                traffic_states=Map.traffic_states,
                traffic_probs=Map.traffic_probs,
                current_traffic_state=self.__current_traffic_state,
                random=self.__random,
            )

            # Lưu trữ kết quả
            self.__segments = segments
            self.__roads = road_info_updated
            self.__intersection_list = list(intersections)
            self.__real_map_ready = True

    def update_real_map(self):
        """
        Cập nhật thông tin của bản đồ thực bao gồm các segment, các điểm giao cắt,
        và thông tin đường, nếu bản đồ chưa được cập nhật.
        """
        if self.__real_map_ready:
            return

        updated_segments, updated_intersections, updated_road_info = (
            self.__real_map.update_segments(
                random=self.__random,
                traffic_states=Map.traffic_states,
                traffic_probs=Map.traffic_probs,
                current_state=self.__current_traffic_state,
            )
        )

        self.__segments = updated_segments
        self.__intersection_list = list(updated_intersections)
        self.__road_info = updated_road_info
        self.__real_map_ready = True

    def save_map_to_file(self, filename="map.txt"):
        """
        Lưu thông tin các đường của bản đồ vào file văn bản.
        Mỗi dòng trong file sẽ là thông tin của một đối tượng Road.
        """
        with open(filename, "w") as file:
            for road in self.__road_info:
                file.write(str(road) + "\n")

    def update_intersections(self):
        """
        Cập nhật danh sách các điểm giao cắt và liên kết các đoạn đường.

        Phương pháp:
        1. Duyệt tất cả các đường và các đường biên.
        2. Tìm giao điểm giữa từng cặp đường khác nhau.
        3. Nếu giao điểm hợp lệ, thêm vào danh sách giao điểm và cập nhật
        danh sách đường đi qua giao điểm đó.
        """
        all_roads = self.__roads + self.__edges
        self.__intersection_list = []

        for road in all_roads:
            for other_road in all_roads:
                if road == other_road:
                    continue

                intersection = road.intersection_point(
                    other_road, self.__map_width, self.__map_height
                )
                self.__intersection_list.append(intersection)

                if intersection != Point(float("inf"), float("inf")):
                    road.add_intersections(intersection)

                    if intersection not in self.__intersections:
                        self.__intersections[intersection] = [road, other_road]
                    else:
                        if road not in self.__intersections[intersection]:
                            self.__intersections[intersection].append(road)
                        if other_road not in self.__intersections[intersection]:
                            self.__intersections[intersection].append(other_road)

    def get_intersections(self):
        """
        Trả về danh sách các điểm giao cắt của tất cả các đoạn đường trong bản đồ.
        """
        return self.__intersection_list

    def draw_map(self):
        """
        Vẽ bản đồ hiện tại và lưu dưới dạng ảnh PNG.

        - Nếu là bản đồ tổng hợp (synthetic), vẽ các đường ngang và dọc.
        - Nếu là bản đồ thực (real map), tải đồ thị từ OSM và vẽ các tuyến đường chính.
        """
        if not map_config["real_map"]:
            for idx, road in enumerate(self.roads):
                x_coords, y_coords = road.get_line().get_points(5000)
                plt.plot(x_coords, y_coords, label=f"road_{idx}")
            plt.xlim(0, 5000)
            plt.ylim(0, 5000)
            plt.savefig("./data/map.png", dpi=300)
            plt.close()
        else:
            custom_filter = '["highway"~"motorway|trunk|primary|secondary|tertiary"]'
            center = map_config["real_center_point"]
            radius = map_config["radius"]
            G = ox.graph_from_point(
                center, dist=radius, custom_filter=custom_filter, network_type="drive"
            )

            # Đặt màu cho tất cả các cạnh
            edge_colors = ["blue" for _ in G.edges()]

            ox.plot_graph(G, edge_color=edge_colors, edge_linewidth=2, node_size=0)
            plt.savefig("./data/map.png", dpi=300)
            plt.close()

    def get_segments(self):
        """Trả về danh sách các đoạn đường (segments) của bản đồ."""
        return self.__segments

    def draw_segments(self):
        """
        Vẽ các đoạn đường trên bản đồ theo trạng thái giao thông.

        - Mỗi trạng thái có màu riêng:
            0: Thông thoáng - Xanh dương
            1: Bình thường - Xanh lá
            2: Trung bình - Vàng
            3: Ùn ứ - Cam
            4: Tắc nghẽn nghiêm trọng - Đỏ
        - Lưu hình ảnh ra file PNG.
        """
        color_map = {
            0: "#52B1FF",  # Thông thoáng
            1: "#33CC33",  # Bình thường
            2: "#FFFF00",  # Trung bình
            3: "#FFA500",  # Ùn ứ
            4: "#FF0000",  # Tắc nghẽn nghiêm trọng
        }

        fig, ax = plt.subplots(figsize=(10, 10))

        if not map_config["real_map"]:
            for state, segments_list in self.__draw_d.items():
                for x_vals, y_vals in segments_list:
                    ax.plot(x_vals, y_vals, color=color_map[state], linewidth=2)
            ax.set_xlim(0, self.__map_width)
            ax.set_ylim(0, self.__map_height)
        else:
            for segment in self.__segments:
                start_point, end_point = segment.get_endpoints()
                x_vals = [start_point.get_point()[0], end_point.get_point()[0]]
                y_vals = [start_point.get_point()[1], end_point.get_point()[1]]
                ax.plot(
                    x_vals, y_vals, color=color_map[segment.get_status()], linewidth=2
                )

        ax.set_title("City Map from Segments")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.savefig(f"./data/current_map_seg{self.__current_traffic_state}.png", dpi=300)
        plt.close()

    def check_valid_next_intersections(self, current_point, visited_points):
        """Trả về các điểm giao nhau tiếp theo hợp lệ từ điểm hiện tại, loại trừ các điểm đã thăm."""
        connected_roads = self.__intersections[current_point]
        next_valid_points = []

        for road in connected_roads:
            intersections = road.get_intersections()
            min_distance = float("inf")
            closest_point = intersections[0]

            for inters in intersections:
                if inters not in visited_points:
                    distance = inters.get_dis_to_point(current_point)
                    if distance < min_distance:
                        min_distance = distance
                        closest_point = inters

            next_valid_points.append(closest_point)

        return next_valid_points
