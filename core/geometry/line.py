import numpy as np

from core.geometry.point import Point



class Line:
    MAX_SLOPE = 1e16  # Ngưỡng giới hạn độ dốc, vượt quá thì coi là vô cực

    """
    Khởi tạo đường thẳng từ điểm và độ dốc hoặc từ hai điểm.

    Args:
        point_1 (Point): Điểm đầu tiên.
        k (float, optional): Độ dốc. Nếu quá lớn hoặc quá nhỏ → ±inf.
        point_2 (Point, optional): Điểm thứ hai, dùng khi slop=False.
        slop (bool, optional): True → dùng k; False → tính từ hai điểm.
    """

    def __init__(self, point_1, k=0, point_2=None, slop=True):
        if slop:
            if abs(k) > Line.MAX_SLOPE or np.isinf(k):
                self.__k = float("inf") if k > 0 else -float("inf")
            else:
                self.__k = k
        else:
            point_2 = point_2 or Point()
            self.__k = point_1.slope_with_point(point_2)

        self.__point = point_1

    def get_slop(self):
        return self.__k

    def get_line(self):
        return (self.__point, self.__k)

    def get_base_point(self):
        return self.__point

    def is_on_line(self, point, eps=1):
        """Kiểm tra xem điểm có nằm trên đường thẳng hay không (sai số cho phép epsilon)."""
        x0, y0 = point.get_point()
        y = self.get_y_at_x(x0)
        if abs(y - y0) < eps:
            return True
        return False

    def get_y_at_x(self, x):
        """Trả về tung độ y tại hoành độ x trên đường thẳng: y = k*(x - x0) + y0."""
        x0, y0 = self.__point.get_point()
        return self.__k * (x - x0) + y0

    def get_line_from_angle(self, angle_deg, point=None):
        """Trả về đường thẳng mới đi qua point, tạo với đường hiện tại một góc 'angle_deg' (độ)."""
        if point is None:
            point = Point()
        angle_rad = np.radians(angle_deg)
        k2 = np.tan(np.arctan(self.__k) - angle_rad)
        # Giới hạn slope để tránh tràn giá trị
        if abs(k2) > Line.MAX_SLOPE:
            k2 = float("inf") if k2 > 0 else -float("inf")
        return Line(point_1=point, k=k2)

    def find_point_at_distance(self, distance, direction="right"):
        """
        Trả về điểm mới cách điểm gốc một khoảng 'distance' trên đường thẳng, theo hướng 'right' (cùng chiều) hoặc 'left' (ngược chiều).
        """
        sign = 1 if direction == "right" else -1
        angle = np.arctan(self.__k)
        dx, dy = sign * np.cos(angle) * distance, sign * np.sin(angle) * distance

        x, y = self.__point.get_point()
        return Point(x + dx, y + dy)

    def get_line_points(self, num_points=100):
        """
        Trả về các tọa độ (x, y) của num_points điểm nằm trên đường thẳng,
        bắt đầu từ điểm gốc và trải dài theo hướng của hệ số góc.
        """
        x_vals, y_vals = [], []
        x0, y0 = self.__point.get_point()

        # Đường thẳng đứng
        if np.isinf(self.__k):
            direction = 1 if self.__k > 0 else -1
            for i in range(1, num_points + 1):
                x_vals.append(x0)
                y_vals.append(y0 + direction * i)
            return x_vals, y_vals

        # Đường thẳng bình thường
        direction = 1 if self.__k >= 0 else -1
        for i in range(1, num_points + 1):
            x_ = x0 + direction * i
            y_ = self.get_y_at_x(x_)
            x_vals.append(x_)
            y_vals.append(y_)
        return x_vals, y_vals

    def get_points(self, num_points=100, step=1):
        """
        Trả về 'num_points' điểm (x, y) nằm trên đường thẳng,
        bắt đầu từ điểm gốc và tăng theo trục X với bước 'step'.
        """
        x0, _ = self.__point.get_point()
        x_vals = [x0 + i * step for i in range(1, num_points + 1)]
        y_vals = [self.get_y_at_x(x) for x in x_vals]
        return x_vals, y_vals

    def intersection_point(self, other_line, max_v, max_h):
        """
        Tính điểm giao giữa hai đường thẳng trong phạm vi [0, max_v] × [0, max_h].
        Trả về Point(inf, inf) nếu không có giao điểm hợp lệ.
        """
        x1, y1 = self.__point.get_point()
        x2, y2 = other_line.get_base_point().get_point()
        k1, k2 = self.__k, other_line.__k

        # Song song → không giao nhau
        if k1 == k2:
            return Point(float("inf"), float("inf"))

        # Trường hợp dựng đứng
        if np.isinf(k1):
            y_int = k2 * (x1 - x2) + y2
            if not (0 <= x1 <= max_v and 0 <= y_int <= max_h):
                return Point(float("inf"), float("inf"))
            return Point(x1, round(y_int, 5))

        if np.isinf(k2):
            y_int = k1 * (x2 - x1) + y1
            if not (0 <= x2 <= max_v and 0 <= y_int <= max_h):
                return Point(float("inf"), float("inf"))
            return Point(x2, round(y_int, 5))

        # Trường hợp bình thường
        x_int = (y2 - y1 + k1 * x1 - k2 * x2) / (k1 - k2)
        y_int = k1 * (x_int - x1) + y1

        # Kiểm tra giới hạn hợp lệ
        if not (0 <= x_int <= max_v and 0 <= y_int <= max_h):
            return Point(float("inf"), float("inf"))

        return Point(round(x_int, 5), round(y_int, 5))
