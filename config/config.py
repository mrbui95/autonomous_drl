map_config = {
    "real_map": True,
    "real_center_point": (
        20.9816147,
        105.7862841,
    ),  # Post and Telecommunications Institute of Technology
    "radius": 2500,
    "num_roads": 15,
    "traffic_level": 1,
    "from_file": 1,
}
"""
Cấu hình bản đồ cho đối tượng Map.

Các tham số:
- is_real_map (bool): True nếu sử dụng bản đồ thực, False nếu tạo bản đồ giả lập.
- center_point (tuple[float, float]): Tọa độ trung tâm bản đồ (vĩ độ, kinh độ).
- map_radius (int): Bán kính vùng bản đồ tính bằng mét.
- num_roads (int): Số lượng tuyến đường trong bản đồ.
- traffic_level (int): Mức độ tắc nghẽn giao thông (0-4).
- load_from_file (bool|int): Nếu True, khởi tạo bản đồ từ file; nếu False, tạo mới.
"""



traffic_profiles = {
    0: {"prob": None, "states": [0], "speed": 20, "task_rate": 20},                                 # Lưu thông tự do
    1: {"prob": [0.95, 0.05], "states": [0, 1], "speed": 17, "task_rate": 30},                      # Ổn định
    2: {"prob": [0.7, 0.25, 0.05], "states": [0, 1, 2], "speed": 14, "task_rate": 40},              # Chậm
    3: {"prob": [0.5, 0.25, 0.2, 0.05], "states": [0, 1, 2, 3], "speed": 10, "task_rate": 50},      # Tắc nghẽn
    4: {"prob": [0.2] * 5, "states": [0, 1, 2, 3, 4], "speed": 5, "task_rate": 60},                 # Tắc nghẽn nghiêm trọng
}
"""
Bảng cấu hình mô tả các trạng thái giao thông và thông số tương ứng.

Mỗi khóa (0–4) biểu thị một cấp độ lưu thông:
    0 - Lưu thông tự do: xe di chuyển tối đa, không cản trở.
    1 - Lưu thông ổn định: tốc độ giảm nhẹ do mật độ tăng.
    2 - Lưu thông chậm: mật độ cao, xe thường xuyên giảm tốc.
    3 - Lưu thông tắc nghẽn: di chuyển chậm, có thể dừng ngắn.
    4 - Tắc nghẽn nghiêm trọng: gần như tê liệt, tốc độ rất thấp.

Các giá trị gồm:
    prob (list[float] | None): Xác suất chọn trạng thái con (None nếu chỉ có 1 trạng thái).
    states (list[int]): Danh sách trạng thái con có thể xảy ra.
    speed (float): Vận tốc trung bình (m/s).
    task_rate (float): Tốc độ sinh tác vụ (tasks/s).
"""

# Hạt giống ngẫu nhiên sinh MEC
SEED_GLOBAL = 42

# Cấu hình dùng GPU số mấy
DEVICE = 1


network_config = {
    "maximum_MECs": 20,
    "cpu_freq_range": [100, 300], # 100 -300 (Hz)
    "best_rate_radius":  100 # 100(m)
}
"""
Cấu hình các tham số cho mạng MEC (Mobile Edge Computing).

Các khóa cấu hình:
    maximum_MECs (int): 
        Số lượng MEC tối đa có thể được triển khai trong mạng.

    cpu_freq_range (list[int]): 
        Dải tần số CPU (tính bằng MHz) mà mỗi MEC có thể được gán ngẫu nhiên.  
        Ví dụ: [100, 300] nghĩa là mỗi MEC sẽ có tần số CPU nằm trong khoảng từ 100Hz đến 300Hz.

    best_rate_radius (int): 
        Bán kính vùng phủ sóng tối ưu (tính bằng mét).  
        Nếu thiết bị nằm trong phạm vi này so với MEC, tốc độ truyền dữ liệu được coi là tốt nhất.
"""


other_config = {
    'apply_thread': 0,
    'apply_detach': 0,
    'score_window_size': 100,
    'tau': 3600 # giây
}


task_config = {
    'data_size_range': [100, 500],     # Kích thước dữ liệu truyền (kB)
    'compute_load_range': [1, 3],      # Khối lượng tính toán (MCycles)
    'task_rate_options': [10, 30, 50], # Các giá trị tốc độ sinh tác vụ (task/s)
    'avg_speed': 10,                   # Vận tốc trung bình của xe (m/s)
    'time_limit': other_config['tau'],                 # Giới hạn thời gian hoàn thành nhiệm vụ (phút)
    'cost_coefficient': 5e-5,          # Hệ số chi phí xử lý tác vụ
    'max_speed': 20                    # Vận tốc cực đại của xe (m/s)
}
"""
Cấu hình các tham số mô phỏng tác vụ (Task Configuration).

Thuộc tính:
-----------
data_size_range : list[int, int]
    Phạm vi kích thước dữ liệu truyền (tính bằng kilobyte - kB).
compute_load_range : list[int, int]
    Phạm vi khối lượng tính toán (tính bằng megacycles - MCycles).
task_rate_options : list[int]
    Các giá trị cường độ sinh tác vụ (task/giây).
avg_speed : float
    Vận tốc trung bình của phương tiện (mét/giây).
time_limit : float
    Giới hạn thời gian hoàn thành nhiệm vụ (phút).
cost_coefficient : float
    Hệ số chi phí xử lý tác vụ, ảnh hưởng đến lợi nhuận cuối cùng.
max_speed : float
    Vận tốc cực đại của phương tiện (mét/giây).
"""


mission_config = {
    'total_missions': 25,          # Tổng số nhiệm vụ cần được phân bổ trong hệ thống.
    'reward_range': [50, 100],     # Khoảng giá trị phần thưởng (hoặc lợi ích) cho mỗi nhiệm vụ.
    'num_vehicles': 5,             # Số lượng phương tiện tham gia thực hiện nhiệm vụ.
    'max_missions_per_vehicle': 5  # Số lượng nhiệm vụ tối đa mà mỗi phương tiện có thể đảm nhận.
}
"""
Cấu hình tổng thể cho hệ thống phân phối nhiệm vụ.

Các tham số:
    total_missions (int): Tổng số nhiệm vụ cần được giao trong hệ thống.
    reward_range (list[int, int]): Khoảng giá trị phần thưởng cho mỗi nhiệm vụ, biểu thị lợi ích tiềm năng.
    num_vehicles (int): Số lượng phương tiện (hoặc tác nhân) tham gia thực hiện nhiệm vụ.
    max_missions_per_vehicle (int): Giới hạn số nhiệm vụ tối đa mà mỗi phương tiện có thể đảm nhận cùng lúc.

Mục đích:
    Cấu hình này được dùng để khởi tạo dữ liệu đầu vào cho mô phỏng hoặc thuật toán 
    phân phối nhiệm vụ giữa các phương tiện trong hệ thống tự hành hoặc hệ thống đa tác nhân.
"""



