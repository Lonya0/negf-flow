import os
from copy import deepcopy
from pathlib import Path
from pathlib import Path
from typing import (
    List,
    Optional,
    Type,
)

from dflow import (
    InputArtifact,
    InputParameter,
    Inputs,
    OutputArtifact,
    OutputParameter,
    Outputs,
    Step,
    Steps,
    argo_len,
    argo_sequence
)
from dflow.python import (
    OP,
    PythonOPTemplate,
    Slices,
)
from dflow.python import OPIO, OPIOSign, Artifact, BigParameter

from negfflow.op.breakdown import Breakdown
from negfflow.op.prep_negf import PrepNegf
from negfflow.op.run_negf import RunNegf
from negfflow.utils.executor import init_executor
from negfflow.utils.pack_files import pack_files, unpack_files
from negfflow.utils.safe_symlink import safe_symlink
from negfflow.utils.set_directory import set_directory


class RunNegfMasterStep(Steps):
    def __init__(self, name, breakdown_config, run_config, upload_python_packages):
        self._input_parameters = {
            "negf_input_config": InputParameter(),
            "task_infos": InputParameter(),
            "task_config": InputParameter()
        }
        self._input_artifacts = {
            "relaxed_systems": InputArtifact(),
            "deeptb_model": InputArtifact()
        }
        self._output_parameters = {
            "task_names": OutputParameter(),
        }
        self._output_artifacts = {
            "logs": OutputArtifact(),
            "negf_results": OutputArtifact(),
            "extra_outputs": OutputArtifact(),
        }

        super().__init__(
            name=name,
            inputs=Inputs(
                parameters=self._input_parameters,
                artifacts=self._input_artifacts,
            ),
            outputs=Outputs(
                parameters=self._output_parameters,
                artifacts=self._output_artifacts,
            ),
        )

        breakdown_op = Breakdown()
        run_op = RunNegf()

        self._keys = ["breakdown", "run-negf"]
        self.step_keys = {}
        self.step_keys["breakdown"] = "breakdown"
        self.step_keys["run-negf"] = "run-negf-{{item}}"

        self = _run_negf_master(
            self,
            self.step_keys,
            breakdown_op,
            run_op,
            breakdown_config=breakdown_config,
            run_config=run_config,
            upload_python_packages=upload_python_packages,
        )

    @property
    def input_parameters(self):
        return self._input_parameters

    @property
    def input_artifacts(self):
        return self._input_artifacts

    @property
    def output_parameters(self):
        return self._output_parameters

    @property
    def output_artifacts(self):
        return self._output_artifacts

    @property
    def keys(self):
        return self._keys

def _run_negf_master(
    steps,
    step_keys,
    breakdown_op: Type[OP],
    run_op: Type[OP],
    breakdown_config: dict,
    run_config: dict,
    upload_python_packages: Optional[List[os.PathLike]] = None,
):
    breakdown_config = deepcopy(breakdown_config)
    run_config = deepcopy(run_config)
    breakdown_template_config = breakdown_config.pop("template_config")
    run_template_config = run_config.pop("template_config")
    breakdown_executor = init_executor(breakdown_config.pop("executor"))
    run_executor = init_executor(run_config.pop("executor"))
    template_slice_config = run_config.pop("template_slice_config", {})

    breakdown = Step(
        "breakdown",
        template=PythonOPTemplate(
            breakdown_op,
            python_packages=upload_python_packages,
            **breakdown_template_config,
        ),
        parameters={},
        artifacts={"tar_file":steps.inputs.artifacts["relaxed_systems"]},
        key=step_keys["prep-negf"],
        executor=prep_executor,
        **prep_config,
    )
    prep_run_steps.add(prep_negf)

    run_negf = Step(
        "run-negf",
        template=PythonOPTemplate(
            run_op,
            slices=Slices(
                "int('{{item}}')",
                input_parameter=["task_name", "modified_negf_input_config"],
                input_artifact=["relaxed_system"],
                output_artifact=["log", "negf_result", "extra_outputs"],
                **template_slice_config,
            ),
            python_packages=upload_python_packages,
            **run_template_config,
        ),
        parameters={
            "task_name": prep_negf.outputs.parameters["task_names"],
            "modified_negf_input_config": prep_negf.outputs.parameters["modified_negf_input_configs"],
        },
        artifacts={
            "relaxed_system": prep_run_steps.inputs.artifacts["relaxed_systems"],
            "deeptb_model": prep_run_steps.inputs.artifacts["deeptb_model"]
        },
        with_sequence=argo_sequence(
            argo_len(prep_negf.outputs.parameters["task_names"])
        ),
        key=step_keys["run-negf"],
        executor=run_executor,
        **run_config
    )
    prep_run_steps.add(run_negf)

    prep_run_steps.outputs.parameters["task_names"].value_from_parameter = prep_negf.outputs.parameters["task_names"]
    prep_run_steps.outputs.artifacts["logs"]._from = run_negf.outputs.artifacts[
        "log"
    ]
    prep_run_steps.outputs.artifacts["negf_results"]._from = run_negf.outputs.artifacts[
        "negf_result"
    ]
    prep_run_steps.outputs.artifacts["extra_outputs"]._from = run_negf.outputs.artifacts[
        "extra_outputs"
    ]

    return prep_run_steps
class RunNegfMaster(OP):

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
            "log": Artifact(List[Path]),
            "extra_outputs": Artifact(List[Path], optional=True),
            "negf_result": Artifact(List[Path]),
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
        from dpnegf.utils.loggers import set_log_handles
        import logging
        from pathlib import Path

        task_name = op_in["task_name"]
        modified_negf_input_config = op_in["modified_negf_input_config"]
        deeptb_model = op_in["deeptb_model"]
        relaxed_system = op_in["relaxed_system"]
        work_dir = Path(task_name)

        relaxed_systems = unpack_files(archive_file_path=relaxed_system,
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
