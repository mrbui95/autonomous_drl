epoch_size = 1000

ddqn_config = {
    "discount_factor":0.95,
    "learning_rate": 1e-5,
    "epsilon": 1.0,
    "epsilon_decay": 0.99,
    "epsilon_min": 0.05,
    "batch_size": 512,
    "maxlen_mem": 10000000,
    "modify_reward": True,
    "combine": 0.00
}
"""
Cấu hình tham số cho agent DDQN.

Các tham số:
- discount_factor (float): Hệ số chiết khấu γ, xác định mức độ ảnh hưởng của phần thưởng tương lai.
- learning_rate (float): Tốc độ học (α) cho optimizer.
- epsilon (float): Xác suất khám phá ban đầu trong epsilon-greedy.
- epsilon_decay (float): Tốc độ giảm epsilon sau mỗi bước huấn luyện.
- epsilon_min (float): Ngưỡng epsilon tối thiểu, đảm bảo vẫn còn xác suất khám phá.
- batch_size (int): Số lượng mẫu mini-batch khi huấn luyện.
- maxlen_mem (int): Kích thước tối đa của replay buffer.
- modify_reward (bool): Cờ xác định có điều chỉnh phần thưởng hay không.
- combine (float): Xác suất sử dụng global memory thay vì local memory khi lấy mini-batch.
"""