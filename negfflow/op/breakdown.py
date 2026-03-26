import os
from pathlib import Path
from typing import List
from dflow.python import OP, OPIO, OPIOSign, Artifact, BigParameter

from negfflow.utils.pack_files import pack_files, unpack_files
from negfflow.utils.set_directory import set_directory
from negfflow.utils.safe_symlink import safe_symlink

class Breakdown(OP):

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "tar_file": Artifact(Path),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "extracted": Artifact(List[Path]),
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        tar_file = op_in["tar_file"]

        work_dir = Path(task_name)

        relaxed_systems = unpack_files(archive_file_path=tar_file,
                                       unpack_dir=work_dir)

        with set_directory(work_dir):
            log_path = 'log'
            log_level = logging.INFO
            set_log_handles(log_level, Path(log_path) if log_path else None)

            log = []
            extra_outputs = []
            negf_result = []

            for sys in relaxed_systems:
                with set_directory(Path(sys.name.replace('.','_'))):
                    safe_symlink(os.path.basename(deeptb_model), deeptb_model)
                    model = build_model(os.path.basename(deeptb_model),
                                        common_options={"device": "cpu"})

                    safe_symlink("relaxed.vasp", sys)

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

                    log.append(Path('.') / "log")
                    extra_outputs.append(pack_files(work_dir=Path('.'),
                                                    file_names=["dos.png","transmission.png","profile_report.html"],
                                                    archive_name="extra_outputs.tar.gz"))
                    negf_result.append(Path('.') / "negf.out.pth")

        op_out = OPIO({
            "log": log,
            "extra_outputs": extra_outputs,
            "negf_result": negf_result,
            "task_name": task_name
        })

        return op_out

    def __init__(self):
        return
