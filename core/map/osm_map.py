import os
import pickle
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

from config.config import map_cfg
from core.geometry.segment import Segment
from core.map.utils import create_segments, get_road_segments

class RealMap:
    """
    Lớp RealMap đại diện cho bản đồ thực tế lấy từ OpenStreetMap (OSM) và xây dựng đồ thị đường phố.
    
    Lớp này:
        - Tải dữ liệu đường phố xung quanh một điểm trung tâm với bán kính xác định.
        - Lọc các cạnh (edges) ngắn và loại bỏ các nút (nodes) cô lập.
        - Giữ lại thành phần liên thông mạnh nhất của đồ thị.
        - Vẽ bản đồ đồ thị và lưu hình ảnh.
        - Chia bản đồ thành các segment, gán trạng thái giao thông, tốc độ trung bình, tốc độ tạo tác vụ.
    
    Attributes:
        G (networkx.MultiDiGraph): Đồ thị đường phố sau xử lý.
        road_segments (dict): Thông tin segment của từng đường.
    """

    def __init__(self, center_point, radius):
        """
        Khởi tạo bản đồ thực tế từ OSM hoặc từ file pickle nếu có.
        
        Args:
            center_point (tuple): Tọa độ trung tâm (lat, long).
            radius (float): Bán kính xung quanh center_point để lấy bản đồ (m).
        """
        map_folder = "map"
        file_path = os.path.join(map_folder, "map.pkl")

        if not map_cfg['from_file']:
            # Lọc các loại đường quan tâm
            custom_filter = '["highway"~"motorway|trunk|primary|secondary|tertiary"]'
            G = ox.graph_from_point(
                center_point, dist=radius, custom_filter=custom_filter, network_type='drive'
            )

            # Loại bỏ các cạnh quá ngắn (<10m)
            short_edges = [
                (u, v, k) for u, v, k, data in G.edges(keys=True, data=True)
                if data.get('length', 0) < 10
            ]
            G.remove_edges_from(short_edges)

            # Loại bỏ các nút cô lập (degree < 2)
            isolated_nodes = [n for n, deg in dict(G.degree()).items() if deg < 2]
            G.remove_nodes_from(isolated_nodes)

            # Giữ lại thành phần liên thông mạnh nhất
            largest_component = max(nx.strongly_connected_components(G), key=len)
            self.G = G.subgraph(largest_component).copy()

            # Tạo thư mục map nếu chưa có
            if not os.path.exists(map_folder):
                os.makedirs(map_folder)
            # Lưu đồ thị vào file
            with open(file_path, 'wb') as f:
                pickle.dump(self.G, f)
        else:
            try:
                # Load đồ thị từ file
                with open(file_path, 'rb') as f:
                    self.G = pickle.load(f)
            except:
                print("Error: Không tìm thấy file map.pkl. Hãy đặt map_cfg['from_file'] = 0 để tải từ OSM.")
                exit(0)

        # Vẽ đồ thị
        fig, ax = ox.plot_graph(
            self.G,
            node_size=10,
            edge_linewidth=0.5,
            bgcolor="lightgray",
            edge_color="black"
        )
        node_positions = {node: (data['x'], data['y']) for node, data in self.G.nodes(data=True)}
        nx.draw_networkx_nodes(
            self.G, node_positions, node_size=10, node_color='red', ax=ax
        )
        plt.savefig("map.png", dpi=300)
        plt.close()

    def build_map(self, traffic_states, traffic_probs, current_state, random):
        """
        Xây dựng segment từ đồ thị và cập nhật trạng thái giao thông cho từng đoạn đường.
        
        Args:
            traffic_states (list): Danh sách trạng thái giao thông (ví dụ: [0,1,2,3,4]).
            traffic_probs (list): Xác suất cho các trạng thái giao thông.
            current_state (int): Trạng thái giao thông hiện tại của bản đồ.
            random (np.random.Generator): Bộ sinh số ngẫu nhiên.
        
        Returns:
            tuple: (segments, road_info_updated, all_intersections)
                - segments: Danh sách các segment trên bản đồ.
                - road_info_updated: Thông tin segment, tốc độ trung bình, tốc độ tác vụ.
                - all_intersections: Tập các giao lộ trên bản đồ.
        """
        self.__road_info = get_road_segments(self.G)
        segments, all_intersections, road_info_updated = self.update_segments(
            random, traffic_states, traffic_probs, current_state
        )
        return segments, road_info_updated, all_intersections

    def update_segments(self, random, traffic_states, traffic_probs, current_traffic_state):
        """
        Cập nhật thông tin segment cho từng đường dựa trên trạng thái giao thông.

        Args:
            random (np.random.Generator): Bộ sinh số ngẫu nhiên.
            traffic_states (list): Danh sách trạng thái giao thông.
            traffic_probs (list): Xác suất các trạng thái.
            current_state (int): Trạng thái giao thông hiện tại.

        Returns:
            tuple: (segments, all_intersections, road_info_updated)
        """
        print("Cập nhật bản đồ...")
        segments = []
        all_intersections = set()
        road_info_updated = {}

        for road, road_segs in self.road_segments.items():
            # Chọn trạng thái ngẫu nhiên cho đường
            road_state = random.choice(traffic_states, p=traffic_probs[current_traffic_state])
            segment_list, avg_task_rate, avg_speed, intersections = create_segments(
                road_state, road_segs, random
            )
            road_info_updated[road] = (segment_list, avg_task_rate, avg_speed)
            all_intersections.update(intersections)
            segments += segment_list

        Segment.segID = 0
        return segments, all_intersections, road_info_updated
    
    

