import abc

class Subject(metaclass=abc.ABCMeta):
    """Lớp trừu tượng đại diện cho đối tượng bị quan sát."""

    @abc.abstractmethod
    def register_observer(self, observer):
        """Đăng ký một observer mới để nhận thông báo."""
        pass

    @abc.abstractmethod
    def remove_observer(self, observer):
        """Xóa một observer khỏi danh sách theo dõi."""
        pass

    @abc.abstractmethod
    def notify_observer(self):
        """Gửi thông báo đến tất cả observer đã đăng ký."""
        pass
