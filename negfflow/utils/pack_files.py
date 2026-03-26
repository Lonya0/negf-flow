from pathlib import Path
import tarfile
from typing import List


def pack_files(work_dir: str | Path,
               file_names: List[str],
               archive_name="archive.tar.gz") -> Path:
    if work_dir.__class__ == str:
        work_dir = Path(work_dir)

    files = [work_dir / file_name for file_name in file_names]

    archive_path = work_dir / archive_name

    with tarfile.open(archive_path, "w:gz") as tar:
        for f in files:
            if f.exists():
                # arcname 控制压缩包内的文件名（避免带完整路径）
                tar.add(f, arcname=f.name)
            else:
                print(f"[Warning] File not found, skip: {f}")

    return archive_path.absolute()

def unpack_files(archive_file_path: str | Path,
                 unpack_dir: str | Path) -> List[Path]:
    archive_file_path = Path(archive_file_path)
    unpack_dir = Path(unpack_dir)

    if not archive_file_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_file_path}")

    unpack_dir.mkdir(parents=True, exist_ok=True)

    extracted_files: List[Path] = []

    def is_within_directory(directory: Path, target: Path) -> bool:
        try:
            target.resolve().relative_to(directory.resolve())
            return True
        except ValueError:
            return False

    with tarfile.open(archive_file_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = unpack_dir / member.name

            # 🚨 防止路径穿越攻击
            if not is_within_directory(unpack_dir, member_path):
                raise Exception(f"Unsafe path detected: {member.name}")

        tar.extractall(path=unpack_dir)

        # 收集所有文件路径（过滤掉目录）
        for member in tar.getmembers():
            if member.isfile():
                extracted_files.append((unpack_dir / member.name).absolute())

    return extracted_files