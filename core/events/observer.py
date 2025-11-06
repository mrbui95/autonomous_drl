import abc

class Observer(metaclass=abc.ABCMeta):
    """Lớp trừu tượng đại diện cho đối tượng quan sát."""

    @abc.abstractmethod
    def update(self, mission):
        """
        Khi một nhiệm vụ hoàn thành, phương thức này sẽ được gọi để
        thông báo cho các observer (ví dụ: xe) rằng nhiệm vụ đã xong.
        """
        pass
