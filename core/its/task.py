class Task:
    """Lớp đại diện cho một tác vụ tính toán trong hệ thống MEC."""

    id_counter = 0  # Biến lớp để tự động tăng ID cho mỗi tác vụ

    def __init__(self, data_size, compute_size, task_id=None):
        """
        Khởi tạo một tác vụ mới.
        
        Args:
            data_size (float): Kích thước dữ liệu đầu vào (MB, KB, ...).
            compute_size (float): Độ phức tạp tính toán (chu kỳ CPU hoặc FLOPs).
            task_id (int, optional): ID của tác vụ. Nếu không truyền, sẽ tự tăng.
        """
        self.__data_size = data_size
        self.__compute_size = compute_size
        self.__id = Task.id_counter if task_id is None else task_id
        Task.id_counter += 1

    def get_data_size(self):
        """Trả về kích thước dữ liệu đầu vào của tác vụ."""
        return self.__data_size

    def get_compute_size(self):
        """Trả về độ phức tạp tính toán của tác vụ."""
        return self.__compute_size

    def get_info(self):
        """Trả về thông tin chi tiết của tác vụ (data_size, compute_size, id)."""
        return (self.__data_size, self.__compute_size, self.__id)

    def __lt__(self, other):
        """So sánh tác vụ hiện tại nhỏ hơn tác vụ khác theo dữ liệu và độ phức tạp."""
        return (self.__data_size < other.__data_size) and (self.__compute_size < other.__compute_size)

    def __eq__(self, other):
        """Hai tác vụ bằng nhau nếu có cùng ID."""
        return self.__id == other.__id
