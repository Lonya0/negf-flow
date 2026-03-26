import time
from pathlib import Path


def wait_for_files(file_paths, max_retries=10, wait_seconds=10):
    """
    等待文件生成（存在且非空）

    Parameters
    ----------
    file_paths : list[Path]
        需要检查的文件路径列表
    max_retries : int
        最大重试次数
    wait_seconds : int
        每次等待时间（秒）
    """

    for attempt in range(max_retries):
        all_ok = True

        for f in file_paths:
            if not f.exists() or f.stat().st_size == 0:
                all_ok = False
                break

        if all_ok:
            print(f"[INFO] 所有文件已生成（第 {attempt+1} 次检查）")
            return True

        print(f"[WARN] 文件尚未就绪，第 {attempt+1}/{max_retries} 次检查...")
        time.sleep(wait_seconds)

    raise RuntimeError("文件在多次等待后仍未生成，请检查上游流程")