import numpy as np

from core.its.mission import Mission
from utils.patterns.observer import Observer

from core.geometry.line import Line
from core.mec.mec_network import MECNetwork

from config.config import task_config


class Vehicle(Observer):
    """
    Lớp đại diện cho một phương tiện trong hệ thống giao thông thông minh (ITS).

    Phương tiện có thể nhận và xử lý các nhiệm vụ (Mission), tương tác với bản đồ,
    và cập nhật trạng thái dựa trên mô hình quan sát (Observer pattern).
    Mỗi phương tiện có khả năng tính toán, di chuyển, và tham gia vào quá trình
    xử lý tác vụ tại MEC hoặc cục bộ.
    """

    vehicle_counter = 0  # Biến đếm toàn cục để gán ID duy nhất cho từng phương tiện

    def __init__(
        self,
        cpu_freq,
        current_position,
        road_map,
        tau=120,
        verbose=False,
        strategy=0,
        non_priority_orders=False,
    ):
        """
        Khởi tạo một đối tượng phương tiện mới trong hệ thống ITS.

        Args:
            cpu_freq (float): Tần số CPU của phương tiện (Hz).
            current_position (Point): Vị trí hiện tại của phương tiện trên bản đồ.
            road_map (Graph): Bản đồ giao thông mà phương tiện di chuyển trên đó.
            tau (int, optional): Thời gian giới hạn cho chu kỳ xử lý nhiệm vụ (phút). Mặc định là 120.
            verbose (bool, optional): Nếu True, in ra log chi tiết hoạt động của phương tiện. Mặc định là False.
            strategy (int, optional): Cờ trạng thái cho biết phương tiện có chọn nhiệm vụ theo từng task hay không. Mặc định là 0.
            non_priority_orders (bool, optional): Nếu True, phương tiện xử lý nhiệm vụ không theo thứ tự ưu tiên. Mặc định là False.
        """
        # Năng lực xử lý của phương tiện (Hz)
        self.__cpu_freq = cpu_freq

        # Vị trí hiện tại của phương tiện trên bản đồ
        self.__current_position = current_position

        # Danh sách nhiệm vụ hiện tại mà phương tiện đang đảm nhận
        self.__missions = []

        # Danh sách nhiệm vụ sẵn sàng để xử lý (không còn phụ thuộc)
        self.__ready_missions = []

        # Danh sách nhiệm vụ đã chấp nhận nhưng chưa bắt đầu
        self.__accepted_missions = []

        # Bản đồ mà phương tiện di chuyển
        self.__road_map = road_map

        # Tổng lợi nhuận tích lũy của phương tiện
        self.__total_profit = 0

        # Số lượng lần bị trễ khi hoàn thành nhiệm vụ
        self.__lateness_count = 0

        # ID duy nhất của phương tiện
        self.__vehicle_id = Vehicle.vehicle_counter
        Vehicle.vehicle_counter += 1

        # Thời gian điều khiển hoặc cập nhật trạng thái gần nhất
        self.__control_time = 0

        # Giới hạn thời gian xử lý nhiệm vụ (giây)
        self.__tau = tau

        # Số nhiệm vụ đã hoàn thành
        self.__completed_count = 0

        # Lợi nhuận hiện tại của phương tiện
        self.__profit = 0

        # Chiến lược xử lý nhiệm vụ (0 = tuần tự, 1 = từng task riêng)
        self.strategy = strategy

        # Biến cờ xác định phương tiện có hoàn thành nhiệm vụ đúng hạn hay không
        self.__on_time = True

        # MEC (Mobile Edge Computing) mà phương tiện đang liên kết
        self.__mec_network = MECNetwork()
        self.__assigned_mec = self.__mec_network.generate_mec_list(road_map.get_intersections())

        # Cờ cho phép xử lý nhiệm vụ không theo thứ tự ưu tiên
        self.__non_priority_orders = non_priority_orders

        # Danh sách thứ tự nhiệm vụ mà phương tiện sẽ thực hiện
        self.__mission_order = []

    def get_vehicle_id(self):
        """Trả về ID của phương tiện."""
        return self.__vehicle_id

    def get_position(self):
        """Lấy vị trí hiện tại của phương tiện."""
        return self.__current_position

    def set_position(self, position):
        """Cập nhật vị trí hiện tại của phương tiện."""
        self.__current_position = position

    @staticmethod
    def reset_vehicle_id_counter():
        """Đặt lại bộ đếm ID cho tất cả phương tiện."""
        Vehicle.vehicle_counter = 0

    def get_control_time(self):
        """Trả về thời gian điều khiển hiện tại của phương tiện."""
        return self.__control_time
    
    def get_accepted_missions(self):
        """Trả về Danh sách nhiệm vụ đã chấp nhận nhưng chưa bắt đầu."""
        return self.__accepted_missions

    def assign_missions(self, solution, mission_list=[], has_order_tuple=False):
        """
        Gán các nhiệm vụ (mission) cho phương tiện dựa trên kết quả giải pháp đầu vào.

        Tham số:
            solution (list | np.ndarray | int): Kết quả ánh xạ nhiệm vụ – có thể là danh sách, mảng numpy hoặc chỉ số.
            mission_list (list): Danh sách tất cả nhiệm vụ có thể gán.
            has_order_tuple (bool): Nếu True, solution chứa cặp (thứ tự, ID phương tiện).

        Trả về:
            bool: True nếu có nhiệm vụ sẵn sàng thực hiện, False nếu không.

        Ngoại lệ:
            ValueError: Nếu loại dữ liệu của solution không hợp lệ.
        """
        if not self.__on_time:
            return

        # Trường hợp 1: solution là danh sách hoặc mảng numpy (mặc định)
        if isinstance(solution, (list, np.ndarray)) and not has_order_tuple:
            for index, vehicle_id in enumerate(solution):
                if vehicle_id == self.__vehicle_id:
                    mission_index = mission_list.index(int(index))
                    mission = mission_list[mission_index]

                    self.accept_mission(mission=mission)

                    if len(mission.get_dependencies()) == 0:
                        mission.update_status(new_status=1)

                    if self.__non_priority_orders and mission.get_status() == 1:
                        self.__ready_missions.append(mission)
                        self.__missions.remove(mission)
                    else:
                        self.check_and_move_ready_mission()

        # Trường hợp 2: solution là chỉ số của mission
        elif isinstance(solution, (int, np.int64)):
            mission_index = mission_list.index(solution)
            mission = mission_list[mission_index]

            if len(mission.get_dependencies()) == 0:
                mission.update_status(new_status=1)

            self.accept_mission(mission=mission)

            if self.__non_priority_orders and mission.get_status() == 1:
                self.__ready_missions.append(mission)
                self.__missions.remove(mission)
            else:
                self.check_and_move_ready_mission()

            return len(self.__ready_missions) > 0

        # Trường hợp 3: solution là danh sách tuple (thứ tự, ID phương tiện)
        elif isinstance(solution, (list, np.ndarray)) and has_order_tuple:
            processed_pairs = []
            for index, (order, vehicle_id) in enumerate(solution):
                if (
                    vehicle_id == self.__vehicle_id
                    and (order, vehicle_id) not in processed_pairs
                ):
                    mission_index = mission_list.index(int(index))
                    mission = mission_list[mission_index]

                    if len(mission.get_dependencies()) == 0:
                        mission.update_status(new_status=1)

                    if self.__non_priority_orders and mission.get_status() == 1:
                        self.__ready_missions.append(mission)
                        self.__missions.remove(mission)
                    else:
                        self.check_and_move_ready_mission()

                    self.accept_mission(mission=mission, order=order)
                    processed_pairs.append((order, vehicle_id))

        else:
            raise ValueError("Giá trị đầu vào 'solution' không hợp lệ.")

    def process_mission_orders(self):
        """
        Xử lý và sắp xếp thứ tự thực hiện nhiệm vụ cho phương tiện.

        Mô tả:
            - Sắp xếp danh sách nhiệm vụ theo thứ tự ưu tiên trong `__orders`.
            - Nếu nhiệm vụ sẵn sàng (status = 1), chuyển sang danh sách nhiệm vụ sẵn sàng.
            - Nếu chưa sẵn sàng, thêm vào danh sách nhiệm vụ đã chấp nhận.
            - Cập nhật lại danh sách nhiệm vụ và nhiệm vụ chấp nhận sau khi xử lý.

        Trả về:
            None
        """
        if len(self.__mission_order) == 0:
            return

        # Sắp xếp danh sách đơn hàng (order_id, priority)
        self.__mission_order.sort(key=lambda order: order[1])

        accepted_missions = []
        is_ready = True

        for order in self.__mission_order:
            order_mission_id = order[0]
            for mission in self.__missions:
                if mission.get_mission_id() == order_mission_id:
                    if mission.get_status() == 1 and is_ready:
                        # Nhiệm vụ sẵn sàng → chuyển sang danh sách ready
                        self.__ready_missions.append(mission)
                        self.__missions.remove(mission)
                        self.__accepted_missions.remove(mission)
                    else:
                        # Nhiệm vụ chưa sẵn sàng → đánh dấu và lưu lại
                        is_ready = False
                        accepted_missions.append(mission)

        # Cập nhật lại danh sách
        self.__missions = accepted_missions
        self.__accepted_missions = accepted_missions.copy()

    def accept_mission(self, mission, order=None):
        """
        Chấp nhận một nhiệm vụ và (nếu có) gán thứ tự thực hiện cho nhiệm vụ đó.

        Mô tả:
            - Đăng ký phương tiện hiện tại làm observer của nhiệm vụ.
            - Thêm nhiệm vụ vào danh sách nhiệm vụ và danh sách nhiệm vụ đã chấp nhận.
            - Nếu có thông tin thứ tự (order), lưu cặp (mission_id, order) vào danh sách thứ tự.

        Tham số:
            mission (Mission): Nhiệm vụ được chấp nhận.
            order (Optional[Any]): Thứ tự hoặc độ ưu tiên của nhiệm vụ (mặc định là None).

        Trả về:
            None
        """
        if order is not None:
            self.__mission_order.append((mission.get_mission_id(), order))

        mission.register_observer(self)
        self.__missions.append(mission)
        self.__accepted_missions.append(mission)

    def update(self, mission, current_time: int = 0):
        """
        Cập nhật trạng thái của phương tiện dựa trên nhiệm vụ đã hoàn thành và thời gian hiện tại.

        Args:
            mission (Mission): Nhiệm vụ vừa được cập nhật trạng thái.
            current_time (int, optional): Thời gian hiện tại trong hệ thống. Mặc định là 0.

        Returns:
            int: Số lượng nhiệm vụ phụ thuộc đã được gỡ bỏ sau khi kiểm tra.
        """
        has_dependency = False
        if len(self.__ready_missions) == 0:
            has_dependency = True

        num_removed_dependencies = self.check_ready(mission=mission)

        if self.__control_time < current_time and has_dependency:
            self.__control_time = current_time

        return num_removed_dependencies

    def check_ready(self, mission):
        """
        Kiểm tra và cập nhật trạng thái sẵn sàng của các nhiệm vụ dựa trên phụ thuộc.
        Trả về số lượng phụ thuộc đã bị xóa.
        """

        if len(self.__missions) == 0:
            return 0

        mission_id = mission.get_mission_id()
        removed_depends_count = 0

        for mis in self.__missions:
            depends = mis.get_dependencies()

            # Nếu nhiệm vụ hiện tại phụ thuộc vào mission_id thì xóa đi
            if mission_id in depends:
                mis.remove_depend_mission(mission_id)
                removed_depends_count += 1
                print(
                    f"Vehicle {self.__vehicle_id} removed dependence {mission_id} from mission {mis.get_mission_id()}"
                )

            # Nếu nhiệm vụ không còn phụ thuộc nào thì cập nhật trạng thái thành sẵn sàng
            if len(depends) == 0:
                mis.update_status(
                    new_status=1, dependent_missions=mission, time=self.__control_time
                )

            # Nếu nhiệm vụ sẵn sàng và danh sách không ưu tiên đang bật thì đưa vào hàng đợi sẵn sàng
            if mis.get_status() == 1 and self.__non_priority_orders:
                self.__ready_missions.append(mis)
                self.__missions.remove(mis)

        # Nếu nhiệm vụ đầu tiên không còn phụ thuộc và chế độ không ưu tiên đang tắt
        if (
            len(self.__missions[0].get_dependencies()) == 0
            and not self.__non_priority_orders
        ):
            mis = self.__missions.pop(0)
            mis.update_status(
                new_status=1, dependent_missions=mission, time=self.__control_time
            )
            if mis not in self.__ready_missions:
                self.__ready_missions.append(mis)

        return removed_depends_count

    def check_and_move_ready_mission(self):
        """
        Kiểm tra và xác nhận nhiệm vụ đầu tiên đã sẵn sàng hay chưa.
        Nếu nhiệm vụ không còn phụ thuộc, chưa có nhiệm vụ nào trong hàng chờ sẵn sàng
        và không ở chế độ không ưu tiên, nhiệm vụ đó sẽ được chuyển sang danh sách sẵn sàng.
        """
        if len(self.__missions) < 1:
            return

        if (
            len(self.__missions[0].get_dependencies()) == 0
            and len(self.__ready_missions) == 0
            and not self.__non_priority_orders
        ):
            # Di chuyển nhiệm vụ đầu tiên sang danh sách nhiệm vụ sẵn sàng
            self.__ready_missions.append(self.__missions[0])

            # Nếu nhiệm vụ này nằm trong danh sách nhiệm vụ đã chấp nhận, thì xóa khỏi đó
            if self.__missions[0] in self.__accepted_missions:
                self.__accepted_missions.remove(self.__missions[0])

            # Xóa nhiệm vụ khỏi danh sách đang xử lý
            self.__missions.remove(self.__missions[0])

    def has_ready_missions(self):
        """
        Kiểm tra xem có nhiệm vụ nào đang trong hàng chờ sẵn sàng hay không.
        Trả về True nếu có, ngược lại False.
        """
        return len(self.__ready_missions) > 0

    def is_on_time(self):
        """Kiểm tra xem phương tiện hoặc nhiệm vụ có đang trong thời gian hợp lệ không."""
        return self.__on_time

    def process_mission(self, missions=None):
        """
        Xử lý nhiệm vụ mà phương tiện đang sẵn sàng thực hiện.
        Ưu tiên thực hiện nhiệm vụ ngắn nhất trước, đồng thời xử lý các nhiệm vụ
        có cùng lộ trình để tối ưu thời gian và lợi nhuận.
        """

        if len(self.__ready_missions) == 0 or not self.__on_time:
            return

        # Lấy nhiệm vụ đầu tiên trong danh sách sẵn sàng
        current_mission = self.__ready_missions.pop(0)

        if len(current_mission.get_dependencies()) != 0:
            raise ValueError("Nhiệm vụ chưa sẵn sàng để xử lý.")

        # --- Bước 1: Tính quãng đường và thời gian di chuyển đến nhiệm vụ ---
        vehicle_pos = self.__current_position
        best_path_to_mission, distance_to_mission = current_mission.get_path_to_start(
            vehicle_pos
        )

        # --- Bước 2: Xử lý các nhiệm vụ có cùng cung đường ---
        main_mission = current_mission
        max_path_len = -float("inf")
        total_profit = current_mission.get_profit()
        total_length = current_mission.get_distance()[0]

        end_points = [current_mission.get_shortest_path()[-1]]
        same_route_missions = [current_mission]

        for mission in list(self.__ready_missions):
            if mission != current_mission and current_mission.is_in_other_road(mission):
                if len(mission.get_shortest_path()) > max_path_len:
                    main_mission = mission
                    max_path_len = len(mission.get_shortest_path())
                total_profit += mission.get_profit()
                total_length += mission.get_distance()[0]
                self.__ready_missions.remove(mission)
                end_points.append(mission.get_shortest_path()[-1])
                same_route_missions.append(mission)

        main_mission.set_profit(total_profit)
        route = main_mission.get_shortest_path()

        # --- Bước 3: Tính toán quãng đường và độ trễ khi di chuyển ---
        road_segments = self.__road_map.get_segments()
        completed_count = 0
        total_delay = distance_to_mission / 10  # vận tốc trung bình: 10 m/s

        if best_path_to_mission:
            best_path_to_mission.pop(-1)
        route = best_path_to_mission + route

        while len(route) > 1:
            start_point = route.pop(0)
            next_point = route[0]
            segment_idx = road_segments.index((start_point, next_point))
            current_segment = road_segments[segment_idx]

            _, avg_speed = current_segment.get_info()
            offload_tasks = current_segment.get_offloading_tasks()
            offload_delay = 0
            on_road_time = start_point.get_dis_to_point(next_point) / avg_speed

            # --- Xử lý offloading task trên đoạn đường ---
            for off_task in offload_tasks:
                line = Line(point_1=start_point, point_2=next_point)
                current_point = start_point

                # Tính tốc độ truyền và năng lực tính toán tại MEC
                rate, mec_cpu = self.__mec_network.get_rate_and_mec_cpu(
                    current_point, self.__assigned_mec
                )
                task_info = off_task.get_info()
                comm_delay = task_info[0] * 8000 / rate
                comp_delay = task_info[1] / mec_cpu
                cur_delay = comm_delay + comp_delay
                offload_delay += cur_delay

                main_mission.update_profit(-task_config["cost_coefficient"])

            total_delay += offload_delay + on_road_time

            # --- Kiểm tra xem có nhiệm vụ nào hoàn thành trên đoạn này không ---
            while start_point in end_points:
                end_points.remove(start_point)
                if self.__control_time + total_delay < self.__tau:
                    completed_count += 1
                else:
                    break

                idx = same_route_missions.index(start_point)
                mission = same_route_missions[idx]
                self.set_position(mission.get_end_point())
                mission.update_status(
                    new_status=2,
                    dependent_missions=missions,
                    time=self.__control_time + total_delay,
                )

                if mission in self.__missions:
                    self.__missions.remove(mission)
                if mission in self.__ready_missions:
                    self.__ready_missions.remove(mission)

                print(f"Nhiệm vụ {mission.get_mission_id()} đã hoàn thành.")

        self.__control_time += total_delay

        # --- Bước 4: Xử lý các nhiệm vụ trễ ---
        if self.__control_time > self.__tau:
            self.__vehicle_profit -= (
                len(self.__missions) + len(self.__ready_missions)
            ) * 50
            self.__missions.clear()
            self.__ready_missions.clear()
            self.__on_time = False
            print("Hết thời gian thực hiện nhiệm vụ.")
            return

        # --- Bước 5: Tính lợi nhuận ---
        total_mis_profit = 0
        while end_points:
            p = end_points.pop(0)
            idx = self.__accepted_missions.index(p)
            total_mis_profit += self.__accepted_missions[idx].get_profit()
            self.__late += 1

        remain_profit = main_mission.get_profit() - total_mis_profit
        remain_profit = max(remain_profit, 0)

        profit = total_length * 0.025 + remain_profit
        self.__profit += remain_profit
        self.__completed_count += completed_count
        self.__vehicle_profit += profit

        # --- Bước 6: Kiểm tra lại thời gian ---
        if self.__control_time > self.__tau:
            self.__missions += self.__ready_missions
            self.__vehicle_profit -= self.calc_profit(self.__missions)
            self.__missions.clear()
            self.__ready_missions.clear()
            self.__on_time = False
            print("Hết thời gian thực hiện nhiệm vụ.")

        return (
            self.__vehicle_id,
            current_mission.get_mission_id(),
            completed_count,
            len(self.__missions),
            profit,
        )

    def reset_total_reward(self):
        """Đặt lại toàn bộ phần thưởng và lợi nhuận của phương tiện."""
        self.__vehicle_profit = 0
        self.__completed_count = 0
        self.__profit = 0

    def calc_profit(self, mission):
        """Tính tổng lợi nhuận từ danh sách hoặc một nhiệm vụ duy nhất."""
        profit = 0
        if isinstance(mission, list):
            for mis in mission:
                profit += mis.get_profit()
        elif isinstance(mission, Mission):
            profit += mission.get_profit()
        return profit

    def get_assigned_missions(self):
        """Trả về danh sách các nhiệm vụ đã được gán cho phương tiện."""
        return self.__missions

    def get_vehicle_profit(self):
        """Trả về tổng lợi nhuận của phương tiện."""
        return self.__vehicle_profit

    def get_total_profit(self):
        """Trả về tổng lợi nhuận kiếm được."""
        return self.__profit

    def get_total_completed(self):
        """Trả về số nhiệm vụ đã hoàn thành."""
        return self.__completed_count
