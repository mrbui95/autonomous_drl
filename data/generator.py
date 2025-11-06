from core.map.map import Map
from core.task_generator import TaskGenerator

from config.config import map_config, task_config, mission_config

def generate(mission_file_name="mission_information.json"):
    map = Map(total_roads=map_config["num_roads"], current_traffic_state=map_config["busy"], from_file=map_config["from_file"])
    tg = TaskGenerator(
        tau=task_config["time_limit"],
        road_map=map,
        min_comp_size=task_config["compute_load_range"][0],
        max_comp_size=task_config["compute_load_range"][1],
        min_data_size=task_config["data_size_range"][0],
        max_data_size=task_config["data_size_range"][1],
    )
    tg.generate_tasks()
    tg.generate_missions(mission_config["total_missions"], file_name=mission_file_name)


if __name__ == "__main__":
    generate()
