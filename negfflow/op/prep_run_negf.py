import os
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from typing import (
    List,
    Optional,
    Set,
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
    Workflow,
    argo_len,
    argo_range,
    argo_sequence,
    download_artifact,
    upload_artifact,
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    OPIOSign,
    PythonOPTemplate,
    Slices,
)
from negfflow.op.prep_negf import PrepNegf
from negfflow.op.run_negf import RunNegf
from negfflow.utils.executor import init_executor

class PrepRunNegf(Steps):
    def __init__(self, name, prep_config, run_config, upload_python_packages):
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

        prep_op = PrepNegf()
        run_op = RunNegf()

        self._keys = ["prep-negf", "run-negf"]
        self.step_keys = {}
        self.step_keys["prep-negf"] = "prep-negf"
        self.step_keys["run-negf"] = "run-negf-{{item}}"

        self = _prep_run_negf(
            self,
            self.step_keys,
            prep_op,
            run_op,
            prep_config=prep_config,
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

def _prep_run_negf(
    prep_run_steps,
    step_keys,
    prep_op: Type[OP],
    run_op: Type[OP],
    prep_config: dict,
    run_config: dict,
    upload_python_packages: Optional[List[os.PathLike]] = None,
):
    prep_config = deepcopy(prep_config)
    run_config = deepcopy(run_config)
    prep_template_config = prep_config.pop("template_config")
    run_template_config = run_config.pop("template_config")
    prep_executor = init_executor(prep_config.pop("executor"))
    run_executor = init_executor(run_config.pop("executor"))
    template_slice_config = run_config.pop("template_slice_config", {})

    prep_negf = Step(
        "prep-negf",
        template=PythonOPTemplate(
            prep_op,
            python_packages=upload_python_packages,
            **prep_template_config,
        ),
        parameters={
            "negf_input_config": prep_run_steps.inputs.parameters["negf_input_config"],
            "task_infos": prep_run_steps.inputs.parameters["task_infos"],
            "task_config": prep_run_steps.inputs.parameters["task_config"]
        },
        artifacts={},
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
    prep_run_steps.outputs.artifacts["logs"]._from = run_negf.outputs.artifacts["log"]
    prep_run_steps.outputs.artifacts["negf_results"]._from = run_negf.outputs.artifacts[
        "negf_result"
    ]
    prep_run_steps.outputs.artifacts["extra_outputs"]._from = run_negf.outputs.artifacts[
        "extra_outputs"
    ]

    return prep_run_steps