import logging
from pathlib import Path
import os.path
from typing import List

from ase import Atoms
from dflow.python import OP, OPIO, OPIOSign, Artifact, BigParameter
from dflow.python.python_op_template import TransientError
from ase.io import read, write

from negfflow.utils.pack_files import pack_files
from negfflow.utils.set_directory import set_directory
from negfflow.utils.run_command import run_command
from negfflow.utils.safe_symlink import safe_symlink
from negfflow.utils.wait_for_files import wait_for_files


class RunLammps(OP):

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "task_path": Artifact(Path),
            "task_name": BigParameter(str),
            "deepmd_model": Artifact(Path),
            "relax_config": dict
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "log": Artifact(Path),
            "extra_outputs": Artifact(Path, optional=True),
            "relaxed_system": Artifact(Path),
            "task_name": BigParameter(str)
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        task_name = op_in["task_name"]
        task_path = op_in["task_path"]
        deepmd_model = op_in["deepmd_model"]
        command = op_in["relax_config"]["run_config"]["command"]
        work_dir = Path(task_name)

        with set_directory(work_dir):
            safe_symlink("in.lammps", task_path / "in.lammps")
            safe_symlink("lammps.data", task_path / "lammps.data")
            safe_symlink(os.path.basename(deepmd_model), deepmd_model)
            command = " ".join([command, "-i", 'in.lammps', "-log", "log.lammps"])
            ret, out, err = run_command(command, shell=True)
            if ret != 0:
                logging.error(
                    "".join(
                        (
                            "lmp failed\n",
                            "command was: ",
                            command,
                            "out msg: ",
                            out,
                            "\n",
                            "err msg: ",
                            err,
                            "\n",
                        )
                    )
                )
                raise TransientError("lmp failed")

            if op_in["relax_config"]["ensemble"] == "smart":
                def build_type_to_element_map(data_file):
                    """
                    从 relaxed.data 构建 type → 元素 映射
                    """
                    atoms = read(data_file, format="lammps-data")

                    types = atoms.arrays["type"]
                    symbols = atoms.get_chemical_symbols()

                    type_map = {}

                    for t, s in zip(types, symbols):
                        if t not in type_map:
                            type_map[t] = s

                    return type_map

                def apply_type_map(_atoms: Atoms, _type_map):
                    """
                    将 LAMMPS type 映射为正确元素
                    """
                    if "type" in atoms.arrays:
                        types = atoms.arrays["type"]
                    elif "types" in atoms.arrays:
                        types = atoms.arrays["types"]
                    else:
                        raise KeyError("未找到 type/types 信息，dump 文件可能格式不对")

                    new_symbols = [_type_map[t] for t in types]
                    _atoms.set_chemical_symbols(new_symbols)

                    return _atoms
                # 构建元素映射关系
                type_map = build_type_to_element_map("relaxed.data")

                # 读取所有结构
                structures = read("traj.xyz", index=":")
                output_paths = []
                for i, atoms in enumerate(structures):
                    atoms = apply_type_map(atoms, type_map)
                    poscar_name = f"POSCAR_{i:04d}.vasp"
                    write(poscar_name, atoms, vasp5=True)
                    output_paths.append(Path(poscar_name))
                wait_for_files(output_paths)
                relaxed_system_paths = output_paths
            else:
                relaxed_system = read("relaxed.data", format='lammps-data')
                write("relaxed.vasp", relaxed_system, vasp5=True)
                relaxed_system_paths = ["relaxed.vasp"]

        relaxed_system_path = pack_files(work_dir=work_dir,
                                         file_names=[Path(path).name for path in relaxed_system_paths],
                                         archive_name="relaxed_system.tar.gz")

        op_out = OPIO({
            "log": Path(work_dir) / "log.lammps",
            "extra_outputs": None,
            "relaxed_system": relaxed_system_path,
            "task_name": task_name
        })

        return op_out

    def __init__(self):
        return
