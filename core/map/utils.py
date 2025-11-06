import copy
import numpy as np

from config.config import traffic_profiles
from geopy.distance import geodesic
from pyproj import Transformer


from core.geometry.point import Point
from core.geometry.segment import Segment

def node_to_point(node):
    """
    Chuyển một node (từ NetworkX) sang đối tượng Point với hệ tọa độ UTM.
    
    Args:
        node (dict): Node từ đồ thị NetworkX chứa 'x' (longitude) và 'y' (latitude).
        
    Returns:
        Point: Điểm trong hệ tọa độ UTM.
    
    Raises:
        ValueError: Nếu tọa độ không hợp lệ.
    """
    wgs84 = 'epsg:4326'  # hệ tọa độ địa lý WGS84
    utm = 'epsg:32647'   # Hệ UTM cho Hà Nội
    transformer = Transformer.from_crs(wgs84, utm, always_xy=True)

    lon = node['x']
    lat = node['y']
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise ValueError("Tọa độ không hợp lệ")
    
    x, y = transformer.transform(lon, lat)
    return Point(x, y)


def calculate_edge_length(u, v, graph):
    """
    Tính chiều dài đường thẳng giữa 2 node trên bản đồ dựa vào tọa độ địa lý.
    
    Args:
        u (int): ID node đầu.
        v (int): ID node cuối.
        graph (nx.Graph): Đồ thị đường phố.
        
    Returns:
        float: Khoảng cách (mét) giữa hai node.
    """
    point_u = (graph.nodes[u]['y'], graph.nodes[u]['x'])
    point_v = (graph.nodes[v]['y'], graph.nodes[v]['x'])
    return geodesic(point_u, point_v).meters


def merge_nearby_nodes(graph, min_distance):
    """
    Gộp các node quá gần nhau để giảm số lượng node và cải thiện hiệu suất tìm kiếm trên bản đồ.
    
    Args:
        graph (nx.Graph): Đồ thị đường phố.
        min_distance (float): Khoảng cách tối thiểu để gộp node (mét).
        
    Returns:
        nx.Graph: Đồ thị đã được gộp node gần nhau.
    """
    nodes_to_remove = []
    merge_dict = {}
    merge_count = 0

    for u, v, data in graph.edges(data=True):
        for u1, v1, data1 in graph.edges(data=True):
            if u1 == u:
                continue

            distance = calculate_edge_length(u, u1, graph)
            if distance < min_distance:
                # Tạo node mới ở trung điểm giữa hai node gần nhau
                new_node_id = len(graph.nodes) + 1
                new_x = (graph.nodes[u]['x'] + graph.nodes[u1]['x']) / 2
                new_y = (graph.nodes[u]['y'] + graph.nodes[u1]['y']) / 2
                edges_to_move = list(graph.edges(u, data=True)) + list(graph.edges(u1, data=True))

                nodes_to_remove.extend([u, u1])
                merge_dict[merge_count] = (new_node_id, new_x, new_y, copy.deepcopy(edges_to_move), distance)
                print(f"Gộp nodes {u}, {u1} (khoảng cách {distance:.2f} m) thành node {new_node_id}")
                merge_count += 1

    # Xóa các node cũ
    graph.remove_nodes_from(nodes_to_remove)

    # Thêm node mới và di chuyển các cạnh
    for merge_key, (new_node_id, new_x, new_y, edges_to_move, _) in merge_dict.items():
        graph.add_node(new_node_id, x=new_x, y=new_y)
        for old_u, old_v, edge_data in edges_to_move:
            if old_v not in nodes_to_remove:
                graph.add_edge(new_node_id, old_v, **edge_data)

    return graph


def get_road_segments(graph):
    """
    Trích xuất thông tin segment của từng đường từ đồ thị đường phố.
    
    Args:
        graph (nx.Graph): Đồ thị đường phố.
        
    Returns:
        dict: {road_name: [(start_point, end_point, length), ...]}
    """
    road_segments = {}
    unnamed_count = 0

    for u, v, data in graph.edges(data=True):
        start_point = node_to_point(graph.nodes[u])
        end_point = node_to_point(graph.nodes[v])

        road_name = data.get('name', None)
        if isinstance(road_name, list):
            road_name = road_name[0]
        if not road_name:
            road_name = f"unknown_{unnamed_count}"

        length = data.get('length', 0)
        segment_info = (start_point, end_point, length)

        if road_name not in road_segments:
            road_segments[road_name] = []
        road_segments[road_name].append(segment_info)
        unnamed_count += 1

    return road_segments

def create_segments(road_state, segment_data, random):
    """
    Tạo các đối tượng Segment từ danh sách các segment thô của một đường,
    gán trạng thái, tốc độ trung bình và tốc độ xử lý công việc dựa trên trạng thái đường.
    
    Args:
        road_state (int): Trạng thái giao thông của đường (0-4).
        segment_data (list): Danh sách các segment thô, mỗi phần tử là tuple 
                            (start_point, end_point, length).
        random (np.random.Generator): Random generator để chọn trạng thái ngẫu nhiên.
    
    Returns:
        tuple: (list_segment_objs, mean_task_rate, mean_speed, set_intersections)
            - list_segment_objs (list[Segment]): Danh sách các đối tượng Segment.
            - mean_task_rate (float): Tốc độ xử lý trung bình của tất cả segment.
            - mean_speed (float): Tốc độ trung bình của tất cả segment (m/s).
            - set_intersections (set): Tập các điểm giao cắt xuất hiện trong segment.
    """
    list_segments = []
    task_rates = []
    speeds = []
    intersections = set()

    profile = traffic_profiles[road_state]

    for start_point, end_point, length in segment_data:
        # Chọn trạng thái ngẫu nhiên nếu có xác suất
        if profile["prob"]:
            segment_status = random.choice(profile["states"], p=profile["prob"])
        else:
            segment_status = profile["states"][0]

        avg_speed = profile["speed"]
        task_rate = profile["task_rate"]

        # Tạo đối tượng Segment
        segment_obj = Segment(
            start_point=start_point,
            end_point=end_point,
            status=segment_status,
            line=None,
            task_rate=task_rate,
            avg_speed=avg_speed,
            distance=length
        )
        list_segments.append(segment_obj)

        # Cập nhật thống kê
        task_rates.append(task_rate)
        speeds.append(avg_speed)
        intersections.update([start_point, end_point])

    mean_task_rate = np.mean(task_rates)
    mean_speed = np.mean(speeds)

    return list_segments, mean_task_rate, mean_speed, intersections

