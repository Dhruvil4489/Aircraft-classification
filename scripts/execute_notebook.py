"""Python script to execute completed notebook end-to-end using nbclient with real outputs."""

import asyncio
import os
import sys
import time
from pathlib import Path

from typing import Any

# Fix for Windows asyncio event loop policy with pyzmq and tornado
if sys.platform == "win32" and sys.version_info < (3, 14):
    if hasattr(asyncio, "WindowsSelectorEventLoopPolicy") and hasattr(asyncio, "set_event_loop_policy"):
        getattr(asyncio, "set_event_loop_policy")(asyncio.WindowsSelectorEventLoopPolicy())

import nbformat
from nbclient import NotebookClient

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
COMPLETED_NB_PATH = WORKSPACE_ROOT / "notebooks" / "completed" / "Final_Project_Classification_and_Captioning_completed.ipynb"
LOG_PATH = WORKSPACE_ROOT / "artifacts" / "logs" / "notebook_execution.log"
DATA_DIR = WORKSPACE_ROOT / "data"


class ProgressNotebookClient(NotebookClient):
    """Custom client that prints cell progress in real time."""
    def execute_cell(self, cell, cell_index, **kwargs):
        src_preview = cell.source[:60].replace("\n", " ") if cell.source else "empty"
        print(f"[Cell {cell_index + 1}/{len(self.nb.cells)}] Running ({cell.cell_type}): {src_preview}...", flush=True)
        t0 = time.time()
        res = super().execute_cell(cell, cell_index, **kwargs)
        duration = time.time() - t0
        print(f" -> Completed in {duration:.1f}s", flush=True)
        return res


def main():
    print(f"Reading notebook from: {COMPLETED_NB_PATH}", flush=True)
    with open(COMPLETED_NB_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Change working directory to DATA_DIR during execution as required by Section K
    os.chdir(DATA_DIR)
    print(f"Kernel execution working directory: {os.getcwd()}", flush=True)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    nb_node: Any = nb
    client = ProgressNotebookClient(
        nb_node,
        timeout=600,
        kernel_name="python3",
        record_timing=True,
    )

    start_time = time.time()
    print("Executing notebook cells with ProgressNotebookClient...", flush=True)

    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        log_file.write(f"=== Notebook Execution Started at {time.ctime()} ===\n")
        log_file.flush()

        try:
            client.execute()
            print("Notebook execution completed successfully!", flush=True)
            log_file.write(f"=== Notebook Execution Succeeded in {time.time() - start_time:.2f} seconds ===\n")
        except Exception as e:
            print(f"Notebook execution failed: {e}", flush=True)
            log_file.write(f"=== Notebook Execution Failed: {e} ===\n")
            log_file.flush()
            raise

    # Write executed notebook back to file
    with open(COMPLETED_NB_PATH, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    # Also sync to Final_Project_Classification_and_Captioning.ipynb under completed
    alt_path = WORKSPACE_ROOT / "notebooks" / "completed" / "Final_Project_Classification_and_Captioning.ipynb"
    with open(alt_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Saved executed notebook with real outputs to {COMPLETED_NB_PATH}", flush=True)
    print(f"Execution log written to {LOG_PATH}", flush=True)


if __name__ == "__main__":
    main()
