from pathlib import Path

from bikewaysim.general_utils import run_notebooks_in_order
if __name__ == "__main__":
    dir = Path.cwd() / "network"
    notebooks = [
        # "step_0_download_process_osm.ipynb",
        # "step_1_network_filtering.ipynb",
        # "step_2_network_reconciliation.ipynb",
        "step_3_add_signals.ipynb",
        # "step_4_bicycle_facilities/step_0_osm_bike_facilities.ipynb",
        # "step_4_bicycle_facilities/step_1_other_bike_facilities.ipynb",
        # "step_4_bicycle_facilities/step_2_bicycle_facilities_reconciliation.ipynb",
        # "step_4_bicycle_facilities/step_3_planned_facilities.ipynb"
        # "step_5_network_modifications.ipynb",
        # "step_6_elevation/step_1_add_elevation.ipynb",
        # "step_6_elevation/step_2_sample_bridge_decks.ipynb",
        # "step_6_elevation/step_3_elevation_cleaning.ipynb",
        # "step_6_elevation/step_4_interpolate_elevation.ipynb",
        "step_7_export_network.ipynb"
    ]
    run_notebooks_in_order(notebooks,dir)