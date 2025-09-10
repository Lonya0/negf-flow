import logging
from dflow.python import OP, OPIOSign, OPIO
import os.path
from dflow.python import OP, OPIO, OPIOSign, Artifact, BigParameter
from dflow.python.python_op_template import TransientError
from pathlib import Path
from ase.io import read, write
from negfflow.utils.set_directory import set_directory
from negfflow.utils.run_command import run_command
from negfflow.utils.safe_symlink import safe_symlink
import os
from dptb.nn.build import build_model
import logging
from pathlib import Path


class RunNegf(OP):

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "modified_negf_input_config": BigParameter(dict),
            "task_name": BigParameter(str),
            "deeptb_model": Artifact(Path),
            "relaxed_system": Artifact(Path)
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "log": Artifact(Path),
            "extra_outputs": Artifact(Path, optional=True),
            "negf_result": Artifact(Path),
            "task_name": BigParameter(str)
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        
        from dpnegf.runner.NEGF import NEGF

        task_name = op_in["task_name"]
        modified_negf_input_config = op_in["modified_negf_input_config"]
        deeptb_model = op_in["deeptb_model"]
        relaxed_system = op_in["relaxed_system"]
        work_dir = Path(task_name)

        with set_directory(work_dir):
            safe_symlink(os.path.basename(deeptb_model), deeptb_model)
            safe_symlink("relaxed.vasp", relaxed_system)
            
            model = build_model(os.path.basename(deeptb_model),
                    common_options={"device":"cpu"})
            negf = NEGF(
                model=model,
                AtomicData_options=modified_negf_input_config['AtomicData_options'],
                structure="relaxed.vasp",
                results_path='.',  
                **modified_negf_input_config['task_options']
            )
            negf.compute()

        op_out = OPIO({
            "log": Path("log"),
            "extra_outputs": None,
            "negf_result": Path("negf.out.pth"),
            "task_name": task_name
        })

        return op_out

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return