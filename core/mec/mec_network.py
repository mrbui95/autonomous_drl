import numpy as np

from config.config import SEED_GLOBAL, network_config
from core.geometry.point import Point
from core.mec.mec import MEC

class MECNetwork:
    """
    Lớp quản lý MEC (Mobile Edge Computing) trong hệ thống tính toán biên di động.
    
    Chức năng:
        - Khởi tạo danh sách MEC với vị trí và tần số CPU ngẫu nhiên.
        - Tính toán tốc độ kênh truyền dựa trên khoảng cách.
        - Xác định MEC tối ưu cho một phương tiện (dựa vào vị trí và năng lực xử lý).
    
    """

    def __init__(self):
        self.__random = np.random.default_rng(SEED_GLOBAL)


    def generate_mec_list(self, intersections):
        """
        Tạo danh sách các MEC (Mobile Edge Computing) bao gồm vị trí và tần số CPU.

        Args:
            intersections (list): Danh sách các vị trí có thể đặt MEC (thường là các giao lộ trong bản đồ mạng).

        Returns:
            list[MEC]: Danh sách các đối tượng MEC, mỗi đối tượng chứa thông tin vị trí và tần số CPU.
        """

        if len(intersections) == 0:
            raise ValueError("Map chưa được khởi tạo — danh sách giao lộ trống.")
        
        # Lấy ngẫu nhiên vị trí MEC
        mec_positions = self.__random.choice(
            intersections,
            network_config['maximum_MECs'],
            replace=False
        )
        
        # Sinh ngẫu nhiên tần số CPU cho từng MEC
        mec_cpu_freq = self.__random.integers(
            network_config['cpu_freq_range'][0],
            network_config['cpu_freq_range'][1],
            network_config['maximum_MECs']
        )

        # Tạo danh sách đối tượng MEC
        mec_list = [MEC(position=position, cpu_freq=freq) for position, freq in zip(mec_positions, mec_cpu_freq)]
        
        return mec_list
    
    def get_channel_rate(self, distance):
        """
        Tính toán tốc độ kênh truyền (channel rate) dựa trên khoảng cách giữa thiết bị phát và thiết bị thu.
        
        Parameters:
            distance (float): Khoảng cách giữa thiết bị phát và thiết bị thu (mét).
        
        Returns:
            float: Giá trị tốc độ kênh truyền (bit/giây).
        
        Notes:
            - Kênh truyền được mô phỏng bằng cách sinh ngẫu nhiên hệ số kênh phức `h`.
            - Công suất phát (Pt) được đặt là 199.526 mW.
            - Mật độ công suất nhiễu (No) là 3.98×10⁻²¹.
            - Số lượng kênh (num_channels) là 10.
            - Băng thông tổng (total_bandwidth) là 20 MHz.
            - Hệ số suy hao đường truyền (path_loss_exp) là 3.
            - Sử dụng định lý Shannon–Hartley để tính tốc độ dữ liệu trung bình trên mỗi kênh.
        """
        
        # Sinh hệ số kênh phức h với phân bố Gaussian
        channel_gain = self.__random.standard_normal(size=(16,)) + 1j * self.__random.standard_normal(size=(16,))
        channel_gain /= np.sqrt(2)  # Chuẩn hóa
        
        # Các tham số truyền
        transmit_power = 199.526e-3           # (W)
        noise_density = 3.98e-21              # (W/Hz)
        num_channels = 10
        total_bandwidth = 20e6                # (Hz)
        path_loss_exp = 3                     # Hệ số suy hao
        
        # Chuẩn hóa hệ số kênh theo khoảng cách
        h_norm = np.linalg.norm(channel_gain, ord=2)
        if round(distance, 2) != 0.0:
            h_norm /= distance ** path_loss_exp
        else:
            h_norm = np.inf  # Tránh chia cho 0
        
        # Băng thông mỗi kênh
        bandwidth_per_channel = total_bandwidth / num_channels
        
        # Tính tốc độ kênh (Shannon capacity)
        rate = bandwidth_per_channel * np.log2(1 + (transmit_power * h_norm) / (bandwidth_per_channel * noise_density))
        
        return rate
    
    def get_rate_and_mec_cpu(self, vehicle_pos, mec_list):
        """
        Xác định tốc độ kênh truyền và công suất CPU của MEC phù hợp nhất cho phương tiện.
        
        Parameters:
            vehicle_pos (Point): Đối tượng vị trí của phương tiện, có phương thức `get_dis_to_point(point)` 
                                để tính khoảng cách tới một điểm bất kỳ.
            mec_list (list[MEC]): Danh sách các đối tượng MEC. 
        
        Returns:
            tuple: (channel_rate, selected_mec_cpu)
                - channel_rate (float): Tốc độ kênh truyền tính được giữa phương tiện và MEC được chọn.
                - selected_mec_cpu (int): Tần số CPU của MEC được chọn (Hz).
        
        Notes:
            - MEC nào nằm trong vùng "best rate radius" (bán kính cho chất lượng kết nối tốt nhất)
            sẽ được ưu tiên lựa chọn.
            - Nếu có nhiều MEC trong vùng này, MEC có `cpu_freq` cao nhất được chọn.
            - Nếu không MEC nào nằm trong vùng tốt, sẽ chọn MEC gần nhất.
            - Hàm `chann_rates()` được gọi để tính tốc độ kênh dựa trên khoảng cách giữa phương tiện và MEC.
        """

        candidate_mecs = []
        min_distance = float("inf") 
        nearest_mec = None

        # Duyệt qua tất cả MEC
        for mec in mec_list:
            mec_position = mec.get_position()
            mec_cpu_freq = mec.get_cpu_freq()

            distance = vehicle_pos.get_dis_to_point(mec_position)

            # Thêm vào danh sách ứng viên nếu trong vùng tốt
            if distance < network_config['best_rate_radius']:
                candidate_mecs.append((mec_position, mec_cpu_freq))

            # Cập nhật MEC gần nhất
            if distance < min_distance:
                min_distance = distance
                nearest_mec = (mec_position, mec_cpu_freq)

        # Chọn MEC phù hợp nhất
        if not candidate_mecs:
            selected_mec = nearest_mec
            selected_distance = min_distance
        else:
            # Chọn MEC có CPU mạnh nhất trong vùng tốt
            selected_mec = max(candidate_mecs, key=lambda m: m[1])
            selected_distance = vehicle_pos.get_dis_to_point(selected_mec[0])

        # Tính tốc độ kênh truyền
        rate = self.get_channel_rate(selected_distance)
        return rate, selected_mec[1]