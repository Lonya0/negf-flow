import os.path
from typing import List
from pathlib import Path
from dflow.python import OP, OPIO, OPIOSign, Artifact, BigParameter
from ase.io import read, write
from ase import Atoms
import numpy as np

from negfflow.utils.safe_symlink import safe_symlink
from negfflow.utils.set_directory import set_directory


class ConvertOnlyOverlap(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "stru_path": Artifact(Path),
            "input_path": Artifact(Path),
            "running_log": Artifact(Path),
            "overlap_input": Artifact(Path)
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "overlap_output": Artifact(Path)
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        stru_cif_path = op_in["stru_path"]
        input_path = op_in["input_path"]
        running_log = op_in["running_log"]
        overlap_input = op_in["overlap_input"]

        import time
        dir_name = "convert_overlap_" + str(time.time())
        work_dir = Path(dir_name)

        with set_directory(work_dir):
            out_abacus_dir = os.path.join(work_dir, "OUT.ABACUS")
            os.makedirs(out_abacus_dir, exist_ok=True)

            with set_directory(out_abacus_dir):
                safe_symlink("STRU.cif", stru_cif_path)
                safe_symlink("INPUT", input_path)
                safe_symlink("running_nscf.log", running_log)
                safe_symlink("data-SR-sparse_SPIN0.csr",overlap_input)

            from dftio.io.parse import ParserRegister

            args = {'command': 'parse', 'log_level': 20, 'log_path': None, 'mode': 'abacus', 'num_workers': 1,
                    'root': '../', 'prefix': dir_name, 'outroot': 'convert_result', 'format': 'dat',
                    'hamiltonian': False, 'overlap': True, 'density_matrix': False, 'eigenvalue': False,
                    'band_index_min': 0, 'energy': False}

            parser = ParserRegister(**args)
            for i in range(len(parser)):
                parser.write(idx=i, **args)

            from pathlib import Path as Pathl

            subdir = next(d for d in Pathl(work_dir / 'convert_result').iterdir() if d.is_dir())


        op_out = OPIO({
            "overlap_output": os.path.join(subdir, "overlaps.h5")
        })

        return op_out
