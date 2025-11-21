import torch as th
import torch.nn as nn

LARGE_VALUE_THRESHOLD = 1e6  # Ngưỡng giá trị lớn, có thể điều chỉnh

class ActorNetwork(nn.Module):
    """
    Mạng Actor cho các thuật toán Reinforcement Learning.

    Attributes:
        state_dim (int): Kích thước vector trạng thái đầu vào.
        hidden_size (int): Số lượng neuron trong các lớp ẩn.
        output_size (int): Kích thước vector đầu ra.
        output_act (callable hoặc nn.Softmax): Hàm kích hoạt cho đầu ra.
        device (torch.device): Thiết bị (CPU/GPU) mà mạng sử dụng.
    
    Methods:
        forward(state):
            Thực hiện forward pass của mạng.
            Kiểm tra giá trị `nan` và các giá trị quá lớn trong input và output.
    """
    
    def __init__(self, state_dim, hidden_size, output_size, output_act, device=th.device("cpu")):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.output_act = output_act
        self.device = device
        
        self.to(self.device)
    
    def forward(self, state):
        """
        Thực hiện forward pass cho trạng thái đầu vào.

        Args:
            state (torch.Tensor): Tensor trạng thái đầu vào có kích thước [batch_size, state_dim].
        
        Returns:
            torch.Tensor: Tensor đầu ra sau khi áp dụng hàm kích hoạt.
        
        Raises:
            ValueError: Nếu input hoặc output chứa giá trị `nan` hoặc giá trị quá lớn.
        """
        state = state.to(self.device)
        
        # Kiểm tra giá trị input
        if th.isnan(state).any():
            raise ValueError("Input chứa giá trị nan.")
        if (state.abs() > LARGE_VALUE_THRESHOLD).any():
            raise ValueError(f"Input chứa giá trị lớn hơn {LARGE_VALUE_THRESHOLD}")
        
        # Forward pass qua các lớp ẩn
        x = nn.functional.relu(self.fc1(state))
        x = nn.functional.relu(self.fc2(x))
        
        # Tính đầu ra với hàm kích hoạt
        if isinstance(self.output_act, nn.Softmax):
            output = self.output_act(self.fc3(x), dim=1)
        elif callable(self.output_act):
            output = self.output_act(self.fc3(x))
        else:
            raise ValueError("output_act phải là callable hoặc nn.Softmax.")
        
        # Kiểm tra giá trị output
        if th.isnan(output).any():
            raise ValueError(f"Output chứa giá trị nan: {output}")
        
        return output
