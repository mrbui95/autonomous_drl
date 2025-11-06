import numpy as np
from collections import deque
import heapq

from core.geometry.point import Point


class Graph:
    def __init__(self, segments):
        """
        Khởi tạo đồ thị từ danh sách các đoạn đường.

        Đây là phương thức khởi tạo của lớp Graph. Nó nhận vào danh sách các
        đối tượng Segment, mỗi Segment thể hiện một đoạn đường nối giữa hai điểm.
        Dựa vào danh sách này, chúng ta xây dựng hai dạng đồ thị:
            1. Đồ thị vô hướng (__undirected_graph): mỗi điểm lưu các đoạn nối đến nó.
            2. Đồ thị có hướng (__directed_graph): mỗi điểm lưu các đoạn đi ra từ nó.

        Args:
            segments (list): Danh sách các đối tượng Segment.
        """
        self.__undirected_graph = {}
        self.__directed_graph = {}

        for s in segments:
            start, end, *_ = s.get_segment()
            self.__undirected_graph.setdefault(start, []).append(s)
            self.__undirected_graph.setdefault(end, []).append(s)
            self.__directed_graph.setdefault(start, []).append(s)

        print("finish build a graph from segment information")

    def get_vertexes(self, directed=False):
        """Trả về các đỉnh của đồ thị; có hướng nếu directed=True."""
        if directed:
            return self.__directed_graph.keys()
        return self.__undirected_graph.keys()
    
    def get_shortest_path(self):
        return self.__shortest_path_info

    def dijkstra(self, start: Point, end: Point, directed: bool = False):
        """
        Tìm đường đi ngắn nhất từ start tới end bằng thuật toán Dijkstra.

        Args:
            start (Point): Điểm bắt đầu.
            end (Point): Điểm kết thúc.
            directed (bool): Nếu True, tính trên đồ thị có hướng; False, vô hướng.

        Returns:
            list[Point]: Danh sách các điểm theo đường đi ngắn nhất.
        """
        graph = self.__undirected_graph
        if start not in graph or end not in graph:
            raise ValueError("Start or end point is not in the graph.")

        INF = float("inf")
        shortest_distance = {v: INF for v in graph}
        previous_vertex = {v: None for v in graph}
        num_hops = {v: INF for v in graph}

        shortest_distance[start] = 0.0
        num_hops[start] = 0
        priority_queue = [(0.0, 0, start)]

        while priority_queue:
            current_distance, current_hops, current_vertex = heapq.heappop(priority_queue)
            # Bỏ qua đỉnh đã cập nhật
            if current_distance > shortest_distance[current_vertex] or current_hops > num_hops[current_vertex]:
                continue
            if current_vertex == end:
                break

            for seg in graph[current_vertex]:
                start_point, end_point, *_ = seg.get_segment()
                if current_vertex not in (start_point, end_point):
                    continue

                neighbors = [end_point] if current_vertex == start_point else []
                if not directed and current_vertex == end_point:
                    neighbors.append(start_point)

                weight = seg.get_distance()
                if weight < 0:
                    raise ValueError(f"Negative weight not supported: {weight}")

                for neighbor_vertex in neighbors:
                    alt_distance = current_distance + weight
                    new_hop = current_hops + 1
                    if alt_distance < shortest_distance[neighbor_vertex] or (alt_distance == shortest_distance[neighbor_vertex] and new_hop < num_hops[neighbor_vertex]):
                        shortest_distance[neighbor_vertex] = alt_distance
                        previous_vertex[neighbor_vertex] = current_vertex
                        num_hops[neighbor_vertex] = new_hop
                        heapq.heappush(priority_queue, (alt_distance, new_hop, neighbor_vertex))

        if shortest_distance[end] == INF:
            raise ValueError("No path exists between the given points.")

        # Tạo danh sách đường đi từ start đến end
        path = []
        node = end
        while node is not None:
            path.insert(0, node)
            node = previous_vertex[node]

        self.__shortest_path_info = (shortest_distance[end], f"{end.get_point()}")
        return path


    def get_graph(self, directed=False):
        """Trả về đồ thị có hướng hoặc vô hướng tùy chọn."""
        return self.__directed_graph if directed else self.__undirected_graph

    def get_possible_roots(self, start_vertex, end_vertex, graph_type="N"):
        """Tìm tất cả các đường đi giữa hai đỉnh trong đồ thị."""
        
        graph = self.__undirected_graph if graph_type == "N" else self.__directed_graph
        queue = deque()
        all_paths = []

        # Khởi tạo hàng đợi từ các đoạn bắt đầu tại start_vertex
        for segment in graph.get(start_vertex, []):
            queue.append((start_vertex, segment, [segment]))

        while queue:
            current_vertex, current_segment, path = queue.popleft()

            st, ed, *_ = current_segment.get_segment()
            next_vertex = ed if current_vertex == st else st

            # Nếu đến đích, lưu đường đi
            if next_vertex == end_vertex and path not in all_paths:
                all_paths.append(path)

            # Thêm các đoạn tiếp theo, tránh vòng lặp
            for next_segment in graph.get(next_vertex, []):
                if next_segment not in path:
                    queue.append((next_vertex, next_segment, path + [next_segment]))

        return all_paths
