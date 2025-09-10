import logging
from pathlib import Path
import os.path
from dflow.python import OP, OPIO, OPIOSign, Artifact, BigParameter
from dflow.python.python_op_template import TransientError
from ase.io import read, write
from negfflow.utils.set_directory import set_directory
from negfflow.utils.run_command import run_command
from negfflow.utils.safe_symlink import safe_symlink

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

            relaxed_system = read("relaxed.data", format='lammps-data')
            write("relaxed.vasp", relaxed_system, vasp5=True)

        op_out = OPIO({
            "log": Path(work_dir) / "log.lammps",
            "extra_outputs": None,
            "relaxed_system": Path(work_dir) / "relaxed.vasp",
            "task_name": task_name
        })

        return op_out

    def __init__(self):
        return
