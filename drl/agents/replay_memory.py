import random
from collections import namedtuple

Experience = namedtuple(
    "Experience", ("state", "action", "reward", "next_state", "done_flag")
)


class ReplayMemory:
    """Bộ nhớ lưu kinh nghiệm (Replay Buffer) cho tác tử RL."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def _add_experience(self, state, action, reward, next_state=None, done_flag=None):
        """Thêm một kinh nghiệm vào bộ nhớ (ghi đè nếu đầy)."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(
            state, action, reward, next_state, done_flag
        )
        self.position = (self.position + 1) % self.capacity

    def add(self, states, actions, rewards, next_states=None, done_flags=None):
        """Thêm một hoặc nhiều kinh nghiệm vào bộ nhớ."""
        if isinstance(states, list):
            if next_states is not None and len(next_states) > 0:
                for s, a, r, n_s, d in zip(
                    states, actions, rewards, next_states, done_flags
                ):
                    self._add_experience(s, a, r, n_s, d)
            else:
                for s, a, r in zip(states, actions, rewards):
                    self._add_experience(s, a, r)
        else:
            self._add_experience(states, actions, rewards, next_states, done_flags)

    def sample(self, batch_size):
        """Lấy mẫu ngẫu nhiên từ bộ nhớ."""
        if batch_size > len(self.memory):
            return False
        transitions = random.sample(self.memory, batch_size)
        return Experience(*zip(*transitions))

    def sample_raw(self, batch_size):
        """Lấy mẫu dạng danh sách (không đóng gói)."""
        if batch_size > len(self.memory):
            return False
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Trả về số lượng kinh nghiệm đang lưu."""
        return len(self.memory)
