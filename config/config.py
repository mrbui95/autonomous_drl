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

map_cfg = {
    "real_map": True,
    "real_center_point": (
        20.9816147,
        105.7862841,
    ),  # Post and Telecommunications Institute of Technology
    "radius": 2500,
    "n_lines": 15,
    "busy": 1,
    "from_file": 0,
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
traffic_profiles = {
    0: {"prob": None, "states": [0], "speed": 20, "task_rate": 20},                                 # Lưu thông tự do
    1: {"prob": [0.95, 0.05], "states": [0, 1], "speed": 17, "task_rate": 30},                      # Ổn định
    2: {"prob": [0.7, 0.25, 0.05], "states": [0, 1, 2], "speed": 14, "task_rate": 40},              # Chậm
    3: {"prob": [0.5, 0.25, 0.2, 0.05], "states": [0, 1, 2, 3], "speed": 10, "task_rate": 50},      # Tắc nghẽn
    4: {"prob": [0.2] * 5, "states": [0, 1, 2, 3, 4], "speed": 5, "task_rate": 60},                 # Tắc nghẽn nghiêm trọng
}

# Hạt giống ngẫu nhiên sinh MEC
SEED_GLOBAL = 42

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
network_cfg = {
    "maximum_MECs": 20,
    "cpu_freq_range": [100, 300], # 100 -300 (Hz)
    "best_rate_radius":  100 # 100(m)
}


