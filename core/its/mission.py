import numpy as np

from utils.patterns.subject import Subject
from core.geometry.point import Point


class Mission(Subject):
    """
    Đại diện cho một nhiệm vụ (Mission) trong hệ thống giao thông thông minh (ITS).

    Mỗi nhiệm vụ mô tả quá trình di chuyển từ một điểm khởi hành (departure point) đến 
    một điểm đích (destination) trong một khoảng thời gian xác định (time slot). 
    Nhiệm vụ có thể phụ thuộc vào các nhiệm vụ khác và được theo dõi bởi các đối tượng 
    quan sát (Observer) như phương tiện (Vehicle). Lớp này tuân theo mô hình thiết kế 
    Observer Pattern — trong đó Mission đóng vai trò là "Subject", có khả năng thông báo 
    đến các "Observer" khi trạng thái thay đổi.

    Thuộc tính lớp:
        mission_counter (int): Biến đếm tĩnh, dùng để gán ID duy nhất cho mỗi nhiệm vụ.
        STATUS (list[str]): Danh sách trạng thái: ["created", "in_progress", "completed"].

    Thuộc tính đối tượng:
        __mission_id (int): ID duy nhất của nhiệm vụ.
        __status (int): Trạng thái hiện tại (0 = created, 1 = in_progress, 2 = completed).
        __start_point (Point): Điểm bắt đầu của nhiệm vụ.
        __end_point (Point): Điểm đích của nhiệm vụ.
        __time_slot (int): Khoảng thời gian mà nhiệm vụ được giao.
        __graph (Graph): Đồ thị biểu diễn mạng đường đi.
        __shortest_path (list[Point]): Tuyến đường ngắn nhất từ điểm bắt đầu đến điểm đích.
        __distance_info (tuple): Thông tin khoảng cách (và các giá trị phụ).
        __dependencies (list[int]): Danh sách ID nhiệm vụ mà nhiệm vụ này phụ thuộc vào.
        __observers (list[Observer]): Danh sách các đối tượng quan sát.
        __profit (float): Lợi nhuận hoặc phần thưởng của nhiệm vụ.

    Phương thức:
        get_end_point(): Trả về điểm đích của nhiệm vụ.
        get_path_to_start(p): Trả về đường đi và khoảng cách từ điểm p đến điểm khởi hành của nhiệm vụ.
        reset(): Đặt lại bộ đếm ID nhiệm vụ về 0.
        get_long(): Lấy thông tin về khoảng cách của nhiệm vụ.
        get_profit(), set_profit(v), update_profit(v): Lấy hoặc cập nhật lợi nhuận.
        get_shortest_path(): Trả về tuyến đường tối ưu.
        is_in_other_road(mission): Kiểm tra xem nhiệm vụ hiện tại có trùng lộ trình với nhiệm vụ khác không.
        get_mid(): Trả về ID nhiệm vụ.

        register_observer(observer_obj): Đăng ký thêm một hoặc nhiều observer.
        remove_observer(observer_obj): Gỡ bỏ một observer cụ thể.
        set_observers(observer_list): Gán lại toàn bộ danh sách observer.
        notify_observer(time): Gửi thông báo đến tất cả observer về thay đổi của nhiệm vụ.

        set_depends(missions), remove_depend(id), get_depends(): Quản lý danh sách các nhiệm vụ phụ thuộc.
        update_status(value, missions=None, time=0): Cập nhật trạng thái nhiệm vụ và thông báo cho các observer hoặc nhiệm vụ phụ thuộc.
        get_status(), print_status(): Lấy hoặc in trạng thái nhiệm vụ.
        get_start_point(), get_end_point(), get_tslot(): Trả về điểm xuất phát, điểm đích hoặc time slot.
        get_trajectory(): Trả về cặp (điểm xuất phát, điểm đích), có thể mở rộng để mô tả quỹ đạo đầy đủ.
    """

    mission_counter = 0
    STATUS  = ["created", "in_progress", "completed"]

    def __init__(self, start_point, end_point, time_slot, graph, verbose=False):
        # Gán ID duy nhất cho nhiệm vụ
        self.__mission_id = Mission.mission_counter

        # Trạng thái ban đầu: 0 (created)
        self.__status = 0

        # Các thuộc tính chính
        self.__start_point = start_point
        self.__end_point = end_point
        self.__time_slot = time_slot
        self.__graph = graph

        # Cập nhật bộ đếm nhiệm vụ toàn cục
        Mission.mission_counter += 1

        # Danh sách phụ thuộc và observer
        self.__dependencies = []
        self.__observers = []

        # Tính toán tuyến đường ngắn nhất và khoảng cách
        self.__shortest_path = self.__graph.dijkstra(start=self.__start_point, end=self.__end_point)
        self.__distance_info = self.__graph.get_shortest_path()

        # Lợi nhuận tạm thời (có thể tinh chỉnh sau)
        self.__profit = 50

    def get_mission_id(self):
        """Trả về mã định danh (ID) của nhiệm vụ."""
        return self.__mission_id

    def get_end_point(self):
        """Trả về điểm đích của nhiệm vụ."""
        return self.__end_point
    
    def set_mission_id(self, val):
        """Gán mã định danh (ID) cho nhiệm vụ."""
        self.__mission_id = val

    def get_path_to_start(self, start_point):
        """Tìm đường đi ngắn nhất và khoảng cách từ vị trí hiện tại đến điểm bắt đầu nhiệm vụ."""
        best_road = self.__graph.dijkstra(start_point, self.__start_point)
        distance = self.__graph.get_shortest_path()[0]
        return best_road, distance
    
    def reset(self):
        """Đặt lại bộ đếm ID của nhiệm vụ."""
        Mission.mission_counter = 0

    def get_distance(self):
        """Trả về độ dài hoặc quãng đường của nhiệm vụ."""
        return self.__distance_info

    def get_profit(self):
        """Trả về lợi nhuận của nhiệm vụ."""
        return self.__profit
    
    def get_shortest_path(self):
        """Trả về đường đi tối ưu nhất của nhiệm vụ."""
        return self.__shortest_path
    
    def set_profit(self, value):
        """Thiết lập lợi nhuận cho nhiệm vụ."""
        self.__profit = value

    def update_profit(self, delta):
        """Cập nhật lợi nhuận bằng cách cộng thêm giá trị mới."""
        self.__profit += delta

    def is_in_other_road(self, mission):
        """Kiểm tra xem đường đi của nhiệm vụ hiện tại có nằm trong đường đi của nhiệm vụ khác hay không."""
        other_road = mission.get_shortest_path()
        for idx, point in enumerate(self.__shortest_path):
            if idx >= len(other_road) or point != other_road[idx]:
                return False
        return True
    
    def __lt__(self, other):
        """So sánh hai nhiệm vụ theo độ dài quãng đường."""
        return self.__distance_info[0] < other.get_distance()[0]

    def __eq__(self, other):
        """Kiểm tra nhiệm vụ có bằng đối tượng khác hay không."""
        if isinstance(other, (int, np.int64)):
            return other == self.__mission_id
        if isinstance(other, Mission):
            return other.get_mission_id() == self.__mission_id
        if isinstance(other, Point):
            return other == self.__end_point
        return False
    
    def update_status(self, new_status, dependent_missions=None, time=0):
        """
        Cập nhật trạng thái của nhiệm vụ và thông báo cho các nhiệm vụ/phương tiện phụ thuộc.

        Args:
            new_status (int): Trạng thái mới của nhiệm vụ (ví dụ: 0 - pending, 1 - ready, 2 - done).
            dependent_missions (list[Mission], optional): Danh sách các nhiệm vụ phụ thuộc vào nhiệm vụ này. Mặc định là None.
            time (int, optional): Thời điểm hiện tại để thông báo observer. Mặc định là 0.

        Returns:
            int: Số lượng phụ thuộc đã bị xoá hoặc observer đã được thông báo.
        """

        self.__status = new_status
        removed_dependencies = 0

        # Nếu nhiệm vụ đã hoàn thành, thông báo cho các nhiệm vụ phụ thuộc
        if Mission.STATUS[new_status] == "completed" and dependent_missions is not None:
            print(f"Nhiệm vụ {self.get_mission_id()} thông báo rằng nó đã hoàn thành để cập nhật các nhiệm vụ phụ thuộc.")
            for mission in dependent_missions:
                dependencies = mission.get_dependencies()
                if self.get_mission_id() in dependencies:
                    dependencies.remove(self.get_mission_id()) 
                    removed_dependencies += 1

        # Khi nhiệm vụ hoàn thành, thông báo cho observer (các phương tiện khác)
        if Mission.status[new_status] == "done":
            removed_dependencies += self.notify_observer(time)
            print(f"Nhiệm vụ {self.get_mission_id()} đã hoàn thành và gửi thông báo đến các phương tiện quan sát.")

        return removed_dependencies


    def get_status(self):
        """Trả về trạng thái hiện tại của nhiệm vụ."""
        return self.__status


    def print_status(self):
        """In ra trạng thái của nhiệm vụ ở dạng chữ."""
        print(Mission.STATUS[self.__status])


    def get_start_point(self):
        """Trả về điểm xuất phát của nhiệm vụ."""
        return self.__start_point


    def get_end_point(self):
        """Trả về điểm đích của nhiệm vụ."""
        return self.__end_point


    def get_route(self):
        """Trả về quỹ đạo (điểm xuất phát, điểm đích) của nhiệm vụ."""
        return (self.__start_point, self.__end_point)


    def get_time_slot(self):
        """Trả về khung thời gian thực hiện nhiệm vụ."""
        return self.__time_slot


    
    # === Nhóm các hàm quản lý observer ===

    def register_observer(self, observers):
        """Đăng ký một hoặc nhiều observer theo dõi nhiệm vụ."""
        if isinstance(observers, list):
            for observer in observers:
                if observer not in self.__observers:
                    self.__observers.append(observer)
        else:
            if observers not in self.__observers:
                self.__observers.append(observers)

    def remove_observer(self, observer):
        """Gỡ bỏ một observer khỏi danh sách theo dõi."""
        if observer in self.__observers:
            self.__observers.remove(observer)

    def set_observers(self, observers):
        """Thiết lập lại toàn bộ danh sách observer."""
        if isinstance(observers, list):
            self.__observers = observers

    def notify_observer(self, time):
        """Gửi thông báo đến các observer và trả về số observer được hoàn thành."""
        removed = 0
        for observer in self.__observers:
            removed += observer.update(self, time)
        return removed
    
    # === Nhóm các hàm xử lý phụ thuộc (dependency) ===
    def update_status(self):
        self.__status = 1 if not self.__dependencies else 0

    def update_depend_mission(self, mission_id):
        """Thêm một nhiệm vụ phụ thuộc và cập nhật trạng thái."""
        self.__dependencies.append(mission_id)
        self.update_status()

    def set_depend_mission(self, missions):
        """Thiết lập danh sách các nhiệm vụ phụ thuộc và cập nhật trạng thái."""
        self.__dependencies = missions
        self.update_status()

    def remove_depend_mission(self, mission_id):
        """Xóa một nhiệm vụ phụ thuộc và cập nhật trạng thái."""
        if mission_id in self.__dependencies:
            self.__dependencies.remove(mission_id)
        self.update_status()

    
    def get_dependencies(self):
        """Trả về danh sách các nhiệm vụ phụ thuộc."""
        return self.__dependencies
    
