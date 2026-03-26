import os
from copy import (
    deepcopy,
)
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

from negfflow.op.dftio import ConvertOnlyOverlap
from negfflow.op.overlap.abacus import AbacusGetOverlap
from negfflow.utils.executor import init_executor


class GetOverlap(Steps):
    def __init__(self, name, get_overlap_config, dftio_config, upload_python_packages):
        self._input_parameters = {
            "run_config": InputParameter()
        }
        self._input_artifacts = {
            "poscar_file": InputArtifact(),
            "input_file": InputArtifact(),
            "pp_files": InputArtifact(),
            "orb_files": InputArtifact()
        }
        self._output_parameters = {}
        self._output_artifacts = {
            "overlap": OutputArtifact(),
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

        get_overlap_op = AbacusGetOverlap()
        convert_only_overlap_op = ConvertOnlyOverlap()

        self._keys = ["get-overlap", "convert-only-overlap"]
        self.step_keys = {}
        self.step_keys["get-overlap"] = "get-overlap-{{item}}"
        self.step_keys["convert-only-overlap"] = "convert-only-overlap-{{item}}"

        self = _get_overlap(
            self,
            self.step_keys,
            get_overlap_op,
            convert_only_overlap_op,
            get_overlap_config=get_overlap_config,
            dftio_config=dftio_config,
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


def _get_overlap(
    steps,
    step_keys,
    get_overlap_op: Type[OP],
    convert_only_overlap_op: Type[OP],
    get_overlap_config: dict,
    dftio_config: dict,
    upload_python_packages: Optional[List[os.PathLike]] = None,
):
    get_overlap_config = deepcopy(get_overlap_config)
    get_overlap_template_config = get_overlap_config.pop("template_config")
    get_overlap_executor = init_executor(get_overlap_config.pop("executor"))
    get_overlap_template_slice_config = get_overlap_config.pop("template_slice_config", {})

    dftio_config = deepcopy(dftio_config)
    dftio_template_config = dftio_config.pop("template_config")
    dftio_executor = init_executor(dftio_config.pop("executor"))
    dftio_template_slice_config = dftio_config.pop("template_slice_config", {})

    get_overlap = Step(
        "get-overlap",
        template=PythonOPTemplate(
            get_overlap_op,
            slices=Slices(
                "int('{{item}}')",
                input_artifact=["poscar_file"],
                output_artifact=["stru_path", "input_path", "running_log", "overlap_output"],
                **get_overlap_template_slice_config,
            ),
            python_packages=upload_python_packages,
            **get_overlap_template_config,
        ),
        parameters={
            "run_config": steps.inputs.parameters["run_config"],
        },
        artifacts={
            "poscar_file": steps.inputs.artifacts["poscar_file"],
            "input_file": steps.inputs.artifacts["input_file"],
            "pp_files": steps.inputs.artifacts["pp_files"],
            "orb_files": steps.inputs.artifacts["orb_files"],
        },
        with_sequence=argo_sequence(
            argo_len(steps.inputs.artifacts["poscar_file"])
        ),
        key=step_keys["get-overlap"],
        executor=get_overlap_executor,
        **get_overlap_config
    )
    steps.add(get_overlap)

    convert_only_overlap = Step(
        "convert-only-overlap",
        template=PythonOPTemplate(
            convert_only_overlap_op,
            slices=Slices(
                "int('{{item}}')",
                input_artifact=["stru_path", "input_path", "running_log", "overlap_input"],
                output_artifact=["overlap_output"],
                **dftio_template_slice_config,
            ),
            python_packages=upload_python_packages,
            **dftio_template_config,
        ),
        artifacts={
            "stru_path": get_overlap.outputs.artifacts["stru_path"],
            "input_path": get_overlap.outputs.artifacts["input_path"],
            "running_log": get_overlap.outputs.artifacts["running_log"],
            "overlap_input": get_overlap.outputs.artifacts["overlap_output"]
        },
        with_sequence=argo_sequence(
            argo_len(get_overlap.outputs.artifacts["stru_path"])
        ),
        key=step_keys["convert-only-overlap"],
        executor=dftio_executor,
        **dftio_config
    )
    steps.add(convert_only_overlap)

    steps.outputs.artifacts["overlap"]._from = convert_only_overlap.outputs.artifacts[
        "overlap_output"
    ]

    return steps
