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
from negfflow.utils.executor import init_executor

class PrepRunRelax(Steps):
    def __init__(self, name, prep_op, run_op, prep_config, run_config, upload_python_packages):
        self._input_parameters = {
            "relax_config": InputParameter(),
            "system_infos": InputParameter(),
            "inputs_config": InputParameter()
        }
        self._input_artifacts = {
            "stacked_systems": InputArtifact(),
            "deepmd_model": InputArtifact()
        }
        self._output_parameters = {
            "task_names": OutputParameter(),
            "task_infos": OutputParameter()
        }
        self._output_artifacts = {
            "logs": OutputArtifact(),
            "relaxed_systems": OutputArtifact(),
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

        self._keys = ["prep-relax", "run-relax"]
        self.step_keys = {}
        self.step_keys["prep-relax"] = "prep-relax"
        self.step_keys["run-relax"] = "run-relax-{{item}}"

        self = _prep_run_relax(
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

def _prep_run_relax(
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

    prep_relax = Step(
        "prep-relax",
        template=PythonOPTemplate(
            prep_op,
            output_artifact_archive={"task_paths": None},
            python_packages=upload_python_packages,
            **prep_template_config,
        ),
        parameters={
            "relax_config": prep_run_steps.inputs.parameters["relax_config"],
            "system_infos": prep_run_steps.inputs.parameters["system_infos"],
            "inputs_config": prep_run_steps.inputs.parameters["inputs_config"]
        },
        artifacts={
            "stacked_systems": prep_run_steps.inputs.artifacts["stacked_systems"],
        },
        key=step_keys["prep-relax"],
        executor=prep_executor,
        **prep_config,
    )
    prep_run_steps.add(prep_relax)

    run_relax = Step(
        "run-relax",
        template=PythonOPTemplate(
            run_op,
            slices=Slices(
                "int('{{item}}')",
                input_parameter=["task_name"],
                input_artifact=["task_path"],
                output_artifact=["log", "relaxed_system", "extra_outputs"],
                **template_slice_config,
            ),
            python_packages=upload_python_packages,
            **run_template_config,
        ),
        parameters={
            "task_name": prep_relax.outputs.parameters["task_names"],
            "relax_config": prep_run_steps.inputs.parameters["relax_config"]
        },
        artifacts={
            "task_path": prep_relax.outputs.artifacts["task_paths"],
            "deepmd_model": prep_run_steps.inputs.artifacts["deepmd_model"]
        },
        with_sequence=argo_sequence(
            argo_len(prep_relax.outputs.parameters["task_names"])
        ),
        key=step_keys["run-relax"],
        executor=run_executor,
        **run_config
    )
    prep_run_steps.add(run_relax)

    prep_run_steps.outputs.parameters["task_names"].value_from_parameter = prep_relax.outputs.parameters["task_names"]
    prep_run_steps.outputs.parameters["task_infos"].value_from_parameter = prep_relax.outputs.parameters["task_infos"]
    prep_run_steps.outputs.artifacts["logs"]._from = run_relax.outputs.artifacts["log"]
    prep_run_steps.outputs.artifacts["relaxed_systems"]._from = run_relax.outputs.artifacts[
        "relaxed_system"
    ]
    prep_run_steps.outputs.artifacts["extra_outputs"]._from = run_relax.outputs.artifacts[
        "extra_outputs"
    ]

    return prep_run_steps