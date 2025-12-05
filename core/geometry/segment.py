import numpy as np
from core.geometry.point import Point


class Segment:
    segment_counter = 0

    """
    Segment đại diện cho một đoạn đường (hoặc kết nối) giữa hai điểm trong hệ thống.

    Mỗi segment nối hai điểm: điểm bắt đầu (__start_point) và điểm kết thúc (end_point), 
    đồng thời lưu thông tin về trạng thái, chiều dài, tốc độ trung bình, 
    và tốc độ tạo tác vụ (task rate). Nếu không chỉ định độ dài, 
    hệ thống sẽ tự tính dựa trên khoảng cách giữa hai điểm, 
    có điều chỉnh theo trạng thái.

    Attributes:
        __start_point (Point): Điểm bắt đầu của đoạn.
        __end_point (Point): Điểm kết thúc của đoạn.
        __status (int): Trạng thái hoặc hệ số ảnh hưởng của đoạn.
        __line (Line | None): Tuyến (Line) chứa đoạn này.
        __sid (int): ID duy nhất của segment.
        __distance (float): Chiều dài đoạn, có thể tính tự động.
        __offloading_tasks (list): Danh sách các tác vụ được offload qua đoạn.
        __task_rate (float | None): Tốc độ tạo tác vụ trên đoạn.
        __avg_speed (float | None): Tốc độ trung bình di chuyển trên đoạn.
    """

    def __init__(
        self,
        start_point,
        end_point,
        status,
        line=None,
        task_rate=None,
        avg_speed=None,
        distance=None,
    ):
        self.__start_point = start_point
        self.__end_point = end_point
        self.__status = status
        self.__line = line
        self.__segment_id = Segment.segment_counter
        if distance != None:
            self.__distance = distance
        else:
            self.__distance = start_point.get_dis_to_point(end_point) * (
                self.__status + 1
            )
        self.__offloading_tasks = []
        self.__task_rate = task_rate
        self.__avg_speed = avg_speed
        Segment.segment_counter += 1

    def get_segment(self):
        return (
            self.__start_point,
            self.__end_point,
            self.__line,
            self.__status,
            self.__segment_id,
        )

    def get_info(self):
        """Trả về Tốc độ tạo tác vụ trên đoạn và vận tốc trung bình của đoạn đường."""
        return (self.__task_rate, self.__avg_speed)

    def set_offloading_tasks(self, v):
        self.__offloading_tasks = v

    def get_offloading_tasks(self):
        return self.__offloading_tasks

    def get_endpoints(self):
        return (self.__start_point, self.__end_point)

    def get_distance(self):
        return self.__distance
    
    def get_avg_speed(self):
        return self.__avg_speed

    def get_status(self):
        return self.__status

    def get_segment_id(self):
        return self.__segment_id

    def reset(self):
        Segment.segment_counter = 0

    def __str__(self):
        return "[" + str(self.__start_point) + "; " + str(self.__end_point) + "]"

    def __eq__(self, other):
        """So sánh hai đoạn đường theo ID, điểm hoặc tuple điểm."""
        if isinstance(other, Segment):
            return self.__segment_id == other.__segment_id
        elif isinstance(other, tuple) and len(other) == 2:
            # So sánh hai điểm đầu-cuối (không phân biệt thứ tự)
            return (
                self.__start_point == other[0] and self.__end_point == other[1]
            ) or (self.__start_point == other[1] and self.__end_point == other[0])
        elif isinstance(other, Point):
            # Kiểm tra điểm có phải là đầu/cuối đoạn
            return other == self.__start_point or other == self.__end_point
        elif isinstance(other, int):
            # So sánh theo ID
            return self.__segment_id == other
        return False

    def get_points(self, step=1):
        """Trả về các điểm (x, y) trên đoạn thẳng với bước nhảy step."""
        x_st, y_st = self.__start_point.get_point()
        x_ed, y_ed = self.__end_point.get_point()

        if x_st == x_ed:  # đường thẳng đứng
            y_vals = np.arange(min(y_st, y_ed), max(y_st, y_ed) + step, step)
            x_vals = [x_st] * len(y_vals)
        else:
            if x_st > x_ed:
                x_st, x_ed = x_ed, x_st  # đảo thứ tự nếu cần
            x_vals = np.arange(x_st, x_ed + step, step)
            y_vals = [self.__line.get_y_at_x(x) for x in x_vals]

        return x_vals, y_vals
