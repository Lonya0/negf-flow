from pathlib import Path

def safe_symlink(link_name, target):
    try:
        Path(link_name).symlink_to(target)
    except FileExistsError:
        print("Link already exists, skipping symlink creation.")
