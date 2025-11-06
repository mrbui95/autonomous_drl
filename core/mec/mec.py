from core.geometry.point import Point


class MEC:
    """ 
    Lớp MEC (Mobile Edge Computing) đại diện cho một máy chủ MEC trong hệ thống tính toán biên di động.
    
    Thuộc tính:
        position (Point): Tọa độ vị trí của MEC trên bản đồ (ví dụ: (x, y)).
        cpu_freq  (int | float): Tần số xử lý CPU của MEC (đơn vị: Hz).
    """
    def __init__(self, position, cpu_freq):
        self.__position = position
        self.__cpu_freq = cpu_freq

    def set_position(self, x, y):
        position = Point(x, y)
        self.__position = position

    def get_position(self):
        return self.__position
    
    def get_cpu_freq(self):
        return self.__cpu_freq
    