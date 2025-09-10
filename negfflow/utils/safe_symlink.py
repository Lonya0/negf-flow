from pathlib import Path
import os
from typing import Optional, Union

def safe_symlink(link_name, target, work_path:Optional[Union[Path, str]]=None):
    try:
        Path(link_name).symlink_to(target)
    except FileExistsError:
        print("Link already exists, skipping symlink creation.")
        pass