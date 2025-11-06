import numpy as np

from core.geometry.segment import Segment
from config.config import traffic_profiles

class Road:
    """
    Lớp `Road` đại diện cho một tuyến đường trong bản đồ.

    Lớp này lưu trữ thông tin về hình học, độ dài, trạng thái,
    các giao lộ liên quan và các đoạn (segment) cấu thành tuyến đường.
    Mỗi đường có một ID duy nhất được tự động gán khi khởi tạo.

    Attributes
    ----------
    next_id : int (class variable)
        Biến tĩnh dùng để sinh ID duy nhất cho mỗi tuyến đường.
    distance_range : list[int, int] (class variable)
        Khoảng giá trị [min, max] độ dài ngẫu nhiên của đường (đơn vị: km).

    id : int
        ID duy nhất của tuyến đường.
    intersections : list
        Danh sách các giao lộ mà tuyến đường đi qua.
    random : numpy.random.Generator
        Bộ sinh số ngẫu nhiên để tạo giá trị độ dài.
    distance : int
        Chiều dài của tuyến đường (km), sinh ngẫu nhiên trong khoảng `length_range_km`.
    geometry_line : Line
        Đối tượng hình học biểu diễn tuyến đường.
    status : int
        Trạng thái hiện tại của đường.
            0 – Tự do: Di chuyển tối đa, không cản trở.
            1 – Ổn định: Tốc độ giảm nhẹ, lưu thông đều.
            2 – Chậm: Mật độ cao, thỉnh thoảng dừng ngắn.
            3 – Tắc nghẽn: Di chuyển chậm hoặc dừng thường xuyên.
            4 – Tê liệt: Gần như đứng yên, tốc độ cực thấp.
    segments : list
        Danh sách các đoạn (segment) cấu thành tuyến đường.
    """

    next_id = 0
    distance_range = [1, 5]  # km

    def __init__(self, geometry_line, status=0):
        """
        Khởi tạo một đối tượng `Road`.

        Parameters
        ----------
        geometry_line : Line
            Đối tượng hình học biểu diễn tuyến đường.
        status : int, optional
            Trạng thái ban đầu của đường (mặc định = 1).
        Notes
        -----
        - `__id` được gán tự động từ biến tĩnh `next_id`.
        - `distance` được sinh ngẫu nhiên trong khoảng `[1, 5]` km.
        - `intersections` và `segments` khởi tạo rỗng.
        """
        self.__id = Road.next_id
        Road.next_id += 1

        self.__random = np.random.default_rng(42)
        self.__distance = self.__random.integers(
            Road.distance_range[0], Road.distance_range[1]
        )
        self.__geometry_line = geometry_line
        self.__status = status
        self.__segments = []
        self.__intersections = []

    def get_avg_speed(self):
        """Trả về tốc độ trung bình của tuyến đường."""
        return self.__avg_speed

    def get_task_rate(self):
        """Trả về tốc độ tạo tác vụ (task rate) trên tuyến đường."""
        return self.__task_rate

    def get_road(self):
        """Trả về tuple chứa (ID, giao lộ, chiều dài) của tuyến đường."""
        return (self.__id, self.__intersections, self.__distance)

    def add_intersections(self, val):
        """Thêm giao lộ mới vào danh sách nếu chưa tồn tại."""
        if val not in self.__intersections:
            self.__intersections.append(val)

    def get_intersections(self):
        """Trả về danh sách các giao lộ của tuyến đường."""
        return self.__intersections

    def get_line(self):
        """Trả về đối tượng đường thẳng (Line) đại diện cho tuyến đường."""
        return self.__geometry_line

    def get_id(self):
        """Trả về mã định danh (ID) của tuyến đường."""
        return self.__id
    
    def get_segments(self):
        """Trả về danh sách các đoạn (segment) cấu thành tuyến đường."""
        return self.__segments

    def get_intersection_point(self, other_road, map_width, map_height):
        """Trả về giao điểm giữa hai tuyến đường trong giới hạn bản đồ."""
        return self.__geometry_line.intersection_point(other_road.get_line(), map_width, map_height)
    
    def update_id(self, val):
        """Cập nhật mã định danh (ID) của tuyến đường."""
        self.__id = val

    def __eq__(self, other):
        """So sánh hai tuyến đường theo ID."""
        return self.__id == other.get_id()

    def __str__(self):
        """Trả về chuỗi biểu diễn tuyến đường gồm ID, tọa độ và độ dốc."""
        point, slope = self.__geometry_line.get_line()
        x, y = point.get_point()
        return f"{self.__id},{x},{y},{slope}"


    def build_segments(self):
        """
        Chia con đường thành các đoạn nhỏ dựa trên các giao điểm và trạng thái hiện tại, 
        sau đó tính tốc độ trung bình và tỷ lệ tác vụ trung bình cho toàn bộ con đường.
        """
        # Sắp xếp danh sách giao điểm theo thứ tự
        intersections = sorted(self.__intersections.copy())
        list_task_rate = []
        list_avg_speed = []

        for i in range(len(intersections) - 1):
            if intersections[i] == intersections[i + 1]:
                continue

            avg_speed, task_rate, status = None, None, None

            # Xác định thông số đoạn đường theo trạng thái giao thông
            profile = traffic_profiles.get(self.__status)

            if profile is None:
                continue
            
            # Chọn trạng thái và thiết lập thông số cho đoạn đường
            avg_speed = profile["speed"]
            task_rate = profile["task_rate"]
            if profile["prob"]:
                status = self.__random.choice(profile["states"], p=profile["prob"])
            else:
                status = profile["states"][0]

            # Lưu thông tin đoạn đường
            list_task_rate.append(task_rate)
            list_avg_speed.append(avg_speed)
            self.__segments.append(
                Segment(intersections[i], intersections[i + 1], status, self.__geometry_line, task_rate, avg_speed)
            )

        # Tính giá trị trung bình của toàn tuyến
        if list_task_rate and list_avg_speed:
            self.__task_rate = np.mean(list_task_rate)
            self.__avg_speed = np.mean(list_avg_speed)
