import torch as th
import torch.nn as nn

LARGE_VALUE_THRESHOLD = 1e6  # Ngưỡng giá trị lớn, có thể điều chỉnh

class ActorCriticNetwork(nn.Module):
    """
    Mạng Actor-Critic, chia sẻ các lớp ẩn dưới nhưng có các lớp đầu ra riêng biệt
    cho Actor và Critic.

    Attributes:
        state_dim (int): Kích thước vector trạng thái đầu vào.
        action_dim (int): Kích thước vector hành động đầu ra của Actor.
        hidden_size (int): Số lượng neuron trong các lớp ẩn.
        actor_output_act (callable hoặc nn.Softmax): Hàm kích hoạt cho đầu ra Actor.
        critic_output_size (int): Kích thước vector đầu ra của Critic.
        device (torch.device): Thiết bị (CPU/GPU) mà mạng sử dụng.

    Methods:
        forward(state):
            Thực hiện forward pass cho trạng thái đầu vào, trả về cả action và value.
    """
    
    def __init__(self, state_dim, action_dim, hidden_size,
                 actor_output_act, critic_output_size=30, device=None):
        super(ActorCriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        self.actor_linear = nn.Linear(hidden_size, action_dim)
        self.critic_linear = nn.Linear(hidden_size, critic_output_size)
        
        self.actor_output_act = actor_output_act
        self.device = device or th.device("cuda" if th.cuda.is_available() else "cpu")
        
        self.to(self.device)

    def forward(self, state):
        """
        Thực hiện forward pass cho trạng thái đầu vào.

        Args:
            state (torch.Tensor): Tensor trạng thái [batch_size, state_dim] hoặc [state_dim].
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: action (tensor) và value (tensor) từ Critic.
        
        Raises:
            ValueError: Nếu input hoặc output chứa giá trị nan hoặc giá trị quá lớn.
        """
        state = state.to(self.device)
        
        # Kiểm tra input
        if th.isnan(state).any():
            raise ValueError("Input chứa giá trị nan.")
        if (state.abs() > LARGE_VALUE_THRESHOLD).any():
            raise ValueError(f"Input chứa giá trị lớn hơn {LARGE_VALUE_THRESHOLD}")
        
        # Forward pass qua các lớp ẩn
        x = nn.functional.relu(self.fc1(state))
        x = nn.functional.relu(self.fc2(x))
        
        # Tính action
        if isinstance(self.actor_output_act, nn.Softmax):
            action = self.actor_output_act(self.actor_linear(x), dim=1)
        elif callable(self.actor_output_act):
            action = self.actor_output_act(self.actor_linear(x))
        else:
            raise ValueError("actor_output_act phải là callable hoặc nn.Softmax.")
        
        # Tính value
        value = self.critic_linear(x)
        
        # Kiểm tra output
        if th.isnan(action).any():
            raise ValueError(f"Action chứa giá trị nan: {action}")
        if th.isnan(value).any():
            raise ValueError(f"Value chứa giá trị nan: {value}")
        
        return action, value
