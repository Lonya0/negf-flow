import logging
from pathlib import Path
import os.path
from typing import List

from dflow.python import OP, OPIO, OPIOSign, Artifact, BigParameter
from dflow.python.python_op_template import TransientError
from ase.io import read, write
from negfflow.utils.set_directory import set_directory
from negfflow.utils.run_command import run_command
from negfflow.utils.safe_symlink import safe_symlink

INPUT_file = """INPUT_PARAMETERS
ntype                   {ntype}
calculation             get_S
basis_type              lcao
"""

class AbacusGetOverlap(OP):

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "poscar_file": Artifact(Path),
            "input_file": Artifact(Path),
            "pp_files": Artifact(List[Path]),
            "orb_files": Artifact(List[Path]),
            "run_config": dict
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "stru_path": Artifact(Path),
            "input_path": Artifact(Path),
            "running_log": Artifact(Path),
            "overlap_output": Artifact(Path)
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        poscar_file = op_in["poscar_file"]
        input_file_path = op_in["input_file"]
        pp_files = op_in["pp_files"]
        orb_files = op_in["orb_files"]
        command = op_in["run_config"]["command"]

        import time
        work_dir = Path("getS_" + str(time.time()))

        with set_directory(work_dir):
            import dpdata
            dpdata.System(poscar_file, fmt="vasp/poscar").to("abacus/stru",
                                                             os.path.join(work_dir, "STRU"),
                                                             pp_file=pp_files,
                                                             numerical_orbital=orb_files)

            safe_symlink("INPUT", input_file_path)
            ret, out, err = run_command(command, shell=True)
            if ret != 0:
                logging.error(
                    "".join(
                        (
                            "abacus failed\n",
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
                raise TransientError("abacus failed")



        op_out = OPIO({
            "stru_path": work_dir / "OUT.ABACUS" / "STRU.cif",
            "input_path": work_dir / "OUT.ABACUS" / "INPUT",
            "running_log": work_dir / "OUT.ABACUS" / "running_nscf.log",
            "overlap_output": work_dir / "OUT.ABACUS" / "data-SR-sparse_SPIN0.csr"
        })

        return op_out

    def __init__(self):
        return
