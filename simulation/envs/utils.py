from config.config import task_config, mission_config

def get_ideal_expected_reward():
    """
    Tính toán phần thưởng trung bình lý tưởng (ideal expected reward) 
    cho một phương tiện (vehicle) trong môi trường mô phỏng.

    Phần thưởng này được ước tính dựa trên các giả định:
        - Phương tiện di chuyển liên tục ở tốc độ tối đa.
        - Hoàn thành nhiều nhiệm vụ nhất có thể trong thời gian quy định (tau).
        - Mỗi nhiệm vụ mang lại lợi ích trung bình cộng của hai giá trị lợi ích biên (benefits).
        - Mỗi nhiệm vụ được cộng thêm một phần thưởng cố định (ở đây là 100).

    Returns:
        float: Tổng phần thưởng trung bình lý tưởng mà một vehicle có thể đạt được.
    """
    # Quãng đường tối đa có thể di chuyển trong một segment (mét)
    # max_speed: tốc độ tối đa (m/s)
    # tau: thời gian hoạt động của một segment (giây)
    max_possible_distance = task_config['max_speed'] * task_config['time_limit']  

    # Lợi ích trung bình của một nhiệm vụ (đơn vị tuỳ định nghĩa, ví dụ điểm thưởng)
    avg_mission_benefit = (mission_config['reward_range'][0] + mission_config['reward_range'][1]) / 2

    # Giả sử độ dài trung bình của mỗi nhiệm vụ là 2000 mét
    avg_mission_length = 2000  

    # Số lượng nhiệm vụ tối đa có thể hoàn thành trong phiên
    max_completed_missions = max_possible_distance / avg_mission_length  

    # Phần thưởng trung bình lý tưởng:
    # Tổng lợi ích trung bình + thưởng cố định 100 cho mỗi nhiệm vụ hoàn thành
    ideal_avg_reward = max_completed_missions * (avg_mission_benefit + 100)

    return ideal_avg_reward