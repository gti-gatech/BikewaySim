import nbformat
from nbconvert import ExecutePreprocessor
import os

def run_notebook(notebook_path, timeout=600, kernel_name='python3'):
    """Runs a Jupyter Notebook without saving the output."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    
    executor = ExecutePreprocessor(timeout=timeout, kernel_name=kernel_name)
    executor.preprocess(notebook, {'metadata': {'path': os.path.dirname(notebook_path)}})
    
    print(f"Executed {notebook_path}")

def run_notebooks_in_order(notebook_list):
    """Runs a list of Jupyter notebooks sequentially."""
    for notebook in notebook_list:
        run_notebook(notebook)

if __name__ == "__main__":
    notebooks = [
        "step_0_download_process_osm.ipynb",
        "step_1_network_filtering.ipynb",
        "step_2_network_reconciliation.ipynb",
        "step_3_add_signals.ipynb"
        "step_4_bicycle.ipynb"
    ]
    run_notebooks_in_order(notebooks)
