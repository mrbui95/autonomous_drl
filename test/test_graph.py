import matplotlib.pyplot as plt
from collections import defaultdict

from core.geometry.point import Point
from core.geometry.segment import Segment
from core.geometry.graph import Graph

# Tạo các điểm
A = Point(0, 0)
B = Point(2, 1)
C = Point(4, 0)
D = Point(3, 3)
E = Point(5, 3)

# Tạo các đoạn đường
segments = [
    Segment(A, B, status=1),
    Segment(B, C, status=2),
    Segment(A, C, status=4),
    Segment(B, D, status=3),
    Segment(C, D, status=5),
    Segment(C, E, status=2),
    Segment(D, E, status=3)
]

# Khởi tạo Graph
graph = Graph(segments)


# Test Dijkstra
shortest_path = graph.dijkstra(A, E)
i = 0
for hop in shortest_path:
    i += 1
    print(str(i) + ": (" + str(hop) + ")")
print("Shortest path from A to D:", shortest_path)
print("Distance info:", graph.get_shortest_path())


# Vẽ đồ thị
plt.figure(figsize=(8,8))
points = [A, B, C, D]


# Vẽ các đoạn
for seg in segments:
    print(seg)
    start, end, *_ = seg.get_segment()
    x_start, y_start = start.get_point()
    x_end, y_end = end.get_point()
    plt.plot([x_start, x_end], [y_start, y_end], "b-o")

# Đánh dấu tên các điểm
for p in points:
    x, y = p.get_point()
    plt.text(x + 0.05, y + 0.05, p.get_point(), fontsize=12, color="red")

# Vẽ đường đi ngắn nhất màu khác
for i in range(len(shortest_path)-1):
    s, e = shortest_path[i], shortest_path[i+1]
    sx, sy = s.get_point()
    ex, ey = e.get_point()
    plt.plot([sx, ex], [sy, ey], "g-o", linewidth=2, markersize=8)

plt.title("Graph Visualization with Shortest Path Highlighted")
plt.grid(True)
# plt.show()

