import inspect
from pathlib import Path


def log_pipeline_level():
    # Get the previous frame (the caller)
    frame = inspect.stack()[1]
    caller_file = frame.filename
    level_name = Path(caller_file).name
    print(f"\n-----> {level_name.upper()} <-----")
    # input("PRESS KEY")
