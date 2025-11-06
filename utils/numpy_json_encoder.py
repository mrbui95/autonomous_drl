import json
import numpy as np

from core.geometry.point import Point
from core.its.task import Task

class NumpyJSONEncoder(json.JSONEncoder):
    """
    Bộ mã hóa JSON tùy chỉnh giúp chuyển đổi các kiểu dữ liệu NumPy 
    và một số đối tượng tùy chỉnh (như Task, Point) sang dạng có thể 
    tuần tự hóa được trong JSON.

    Cụ thể:
    - np.integer  → int
    - np.floating → float
    - np.ndarray  → list
    - Task, Point → list thông qua các hàm get_task() hoặc get_point()

    Cách dùng:
        json.dumps(data, cls=NumpyJSONEncoder)
    """

    def default(self, obj):
        """Chuyển đổi các đối tượng không hỗ trợ mặc định sang kiểu JSON hợp lệ."""
        
        # Chuyển kiểu số nguyên của NumPy sang int
        if isinstance(obj, np.integer):
            return int(obj)
        
        # Chuyển kiểu số thực của NumPy sang float
        if isinstance(obj, np.floating):
            return float(obj)
        
        # Chuyển mảng NumPy sang list
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Chuyển đối tượng Task sang list thông qua get_task()
        if isinstance(obj, Task):
            return list(obj.get_info())
        
        # Chuyển đối tượng Point sang list thông qua get_point()
        if isinstance(obj, Point):
            return list(obj.get_point())
        
        # Mặc định: sử dụng phương thức gốc của JSONEncoder
        return super().default(obj)
