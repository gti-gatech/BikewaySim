from pathlib import Path

from bikewaysim.general_utils import run_notebooks_in_order

if __name__ == "__main__":
    dir = Path.cwd() / "map_matching"
    notebooks = [
        "step_0_map_matching_network.ipynb",
        "step_1_map_matching.py",
        "step_2_select_matches.ipynb",
        "step_3_aggregate_trips_to_network.ipynb"
        ]
    run_notebooks_in_order(notebooks,dir)