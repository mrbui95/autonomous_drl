import torch as th
import torch.nn as nn

class CriticNetwork(nn.Module):
    """
    Mạng Critic cho các thuật toán Reinforcement Learning.

    Attributes:
        state_dim (int): Kích thước vector trạng thái đầu vào.
        action_dim (int): Kích thước vector hành động đầu vào.
        hidden_size (int): Số lượng neuron trong các lớp ẩn.
        output_size (int): Kích thước vector đầu ra.
        device (torch.device): Thiết bị (CPU/GPU) mà mạng sử dụng.

    Methods:
        forward(state, action):
            Thực hiện forward pass với trạng thái và hành động đầu vào.
            Hỗ trợ các input có kích thước khác nhau (1D, 2D, 3D).
    """
    
    def __init__(self, state_dim, action_dim, hidden_size, output_size=30, device=None):
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size + action_dim, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        self.device = device or th.device("cuda" if th.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, action):
        """
        Thực hiện forward pass cho trạng thái và hành động đầu vào.

        Args:
            state (torch.Tensor): Tensor trạng thái [batch_size, state_dim] hoặc [state_dim].
            action (torch.Tensor): Tensor hành động [batch_size, action_dim] hoặc [action_dim] hoặc [batch_size, seq_len, action_dim].
        
        Returns:
            torch.Tensor: Tensor giá trị Q của Critic.
        """
        state = state.to(self.device)
        action = action.to(self.device)
        
        # Forward pass qua lớp ẩn đầu tiên
        x = nn.functional.relu(self.fc1(state))
        
        # Điều chỉnh kích thước nếu input là 1D
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        
        # Mở rộng x nếu action có 3 chiều
        if x.dim() == 2 and action.dim() == 3:
            x = x.unsqueeze(1).expand_as(action)
        
        # Nối state và action
        x = th.cat([x, action], dim=-1)
        
        # Forward pass qua lớp ẩn thứ hai
        x = nn.functional.relu(self.fc2(x))
        
        # Lớp đầu ra
        q_value = self.fc3(x)
        
        return q_value
