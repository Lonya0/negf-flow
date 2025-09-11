import os
from pathlib import Path
from typing import List
from dflow.python import OP, OPIO, OPIOSign, Artifact, BigParameter
from negfflow.utils.set_directory import set_directory
from negfflow.utils.safe_symlink import safe_symlink

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
            "extra_outputs": Artifact(List[Path], optional=True),
            "negf_result": Artifact(Path),
            "task_name": BigParameter(str)
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        from dpnegf.runner.NEGF import NEGF
        from dptb.nn.build import build_model
        import torch
        import matplotlib.pyplot as plt

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
            atomic_data_options = None if 'AtomicData_options' in modified_negf_input_config else modified_negf_input_config.get('AtomicData_options')
            negf = NEGF(
                model=model,
                AtomicData_options=atomic_data_options,
                structure="relaxed.vasp",
                results_path='.',
                **modified_negf_input_config['task_options']
            )
            negf.compute()

            negf_out = torch.load('negf.out.pth')

            plt.plot(negf_out['uni_grid'], negf_out['DOS'][str(negf_out['k'][0])])
            plt.xlabel('Energy (eV)')
            plt.ylabel('DOS')
            plt.title('DOS vs Energy')
            plt.grid()
            plt.savefig('dos.png')

            plt.close()

            plt.plot(negf_out['uni_grid'], negf_out['T_avg'])
            plt.xlabel('Energy (eV)')
            plt.ylabel('Transmission')
            plt.title('Transmission vs Energy')
            plt.grid()
            plt.savefig('transmission.png')

        op_out = OPIO({
            "extra_outputs": [Path(work_dir) / "dos.png", 
                              Path(work_dir) / "transmission.png"],
            "negf_result": Path(work_dir) / "negf.out.pth",
            "task_name": task_name
        })

        return op_out

    def __init__(self):
        return
