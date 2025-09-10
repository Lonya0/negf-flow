import json
import os
from copy import deepcopy
from typing import Type, Optional, List

import dflow
import dpdata
from dflow import Step, Steps, InputParameter, InputArtifact, OutputArtifact, Inputs, Outputs, OPTemplate
from dflow.python import OP, PythonOPTemplate
import fpop

import negfflow
from negfflow.op.build_supercell import BuildSupercell
from negfflow.op.overlap import overlap_styles
from negfflow.op.prep_run_negf import PrepRunNegf, PrepNegf, RunNegf
from negfflow.op.prep_run_overlap import PrepRunOverlap
from negfflow.op.prep_run_relax import PrepRunRelax
from negfflow.op.relax import relax_styles
from negfflow.utils.artifact import get_artifact_from_uri, upload_artifact_and_print_uri
from negfflow.utils.executor import init_executor


def make_negf_step(config):
    default_config = config["default_step_config"]

    upload_python_packages = []
    if custom_packages := config.get("upload_python_packages"):
        upload_python_packages.extend(custom_packages)
    upload_python_packages.extend(list(dpdata.__path__))
    upload_python_packages.extend(list(dflow.__path__))
    upload_python_packages.extend(list(fpop.__path__))
    upload_python_packages.extend(list(negfflow.__path__))

    overlap_style = config['overlap']['type'] if config['task']['use_external_overlap'] else None

    ## load parameters
    # negf input config
    if isinstance(config['negf']['config'], str):
        with open(config['negf']['config'], "r") as f:
            negf_input_config = json.load(f)
    elif isinstance(config['negf']['config'], dict):
        negf_input_config = config['negf']['config']
    else:
        negf_input_config = {}

    # relax input config
    if isinstance(config['relax']['config'], str):
        if 'timestep' in config['relax']['config']:
            relax_input_config = config['relax']['config']
        else:
            with open(config['negf']['config'], "r") as f:
                relax_input_config = f.read()
    else:
        relax_input_config = ''

    # overlap input config
    if overlap_style == 'abacus':
        if config["overlap"]["inputs_config"]["input_file"] is not None:
            with open(config["overlap"]["inputs_config"]["input_file"], "r") as f:
                overlap_input_config = f.read()
        else:
            raise FileNotFoundError("pp and orb file must exist!")

    """        """
    parameters = {
        "negf_input_config": negf_input_config,
        "relax_input_config": relax_input_config,
        "overlap_input_config": overlap_input_config,
        "negf_config": config['negf'],
        "task_config": config['task'],
        "relax_config": config['relax'],
        "inputs_config": config['inputs'],
        "overlap_config": config['overlap']
    }

    ## load artifacts
    # init_confs
    if config["inputs"]["init_confs_uri"] is not None:
        init_confs = get_artifact_from_uri(
            config["input"]["init_confs_uri"]
        )
    elif config["inputs"]["init_confs_paths"] is not None:
        init_confs_prefix = config["inputs"]["init_confs_prefix"]
        init_confs = config["inputs"]["init_confs_paths"]
        init_confs = get_systems_from_data(init_confs, init_confs_prefix)
        init_confs = upload_artifact_and_print_uri(init_confs, "init_confs")
    else:
        raise RuntimeError("init_confs must be provided")

    # deepmd model
    if config["inputs"]["deepmd_model_uri"] is not None:
        print("Using uploaded deepmd model at: ", config["inputs"]["deepmd_model_uri"])
        deepmd_model = get_artifact_from_uri(config["inputs"]["deepmd_model_uri"])
    elif config["inputs"]["deepmd_model_path"] is not None:
        deepmd_model = upload_artifact_and_print_uri(
            config["inputs"]["deepmd_model_path"], "deepmd_model"
        )
    else:
        raise FileNotFoundError("deepmd model must exist!")

    # deeptb model
    if config["inputs"]["deeptb_model_uri"] is not None:
        print("Using uploaded deeptb model at: ", config["inputs"]["deeptb_model_uri"])
        deeptb_model = get_artifact_from_uri(config["inputs"]["deeptb_model_uri"])
    elif config["inputs"]["deeptb_model_path"] is not None:
        deeptb_model = upload_artifact_and_print_uri(
            config["inputs"]["deeptb_model_path"], "deeptb_model"
        )
    else:
        raise FileNotFoundError("deeptb model must exist!")

    artifacts = {
            "init_confs": init_confs,
            "deepmd_model": deepmd_model,
            "deeptb_model": deeptb_model
        }

    if overlap_style == 'abacus':
        if (config["overlap"]["inputs_config"]["pp_files"] is not None and
                config["overlap"]["inputs_config"]["orb_files"] is not None):
            pp_orb_list = (list(config["overlap"]["inputs_config"]["pp_files"].values()) +
                           list(config["overlap"]["inputs_config"]["orb_files"].values()))
            pp_orb = upload_artifact_and_print_uri(
                pp_orb_list, "pp_orb"
            )
        else:
            raise FileNotFoundError("pp and orb file must exist!")
        artifacts['pp_orb'] = pp_orb

    negf_op = make_negf_op(
        relax_style=config['relax']['type'],
        overlap_style=overlap_style,
        build_supercell_step_config=config["step_configs"].get("build_supercell_step_config", default_config),
        prep_relax_config=config["step_configs"].get("prep_relax_config", default_config),
        prep_negf_config = config["step_configs"].get("prep_negf_config", default_config),
        prep_overlap_config = config["step_configs"].get("prep_overlap_config", default_config),
        run_relax_config = config["step_configs"].get("run_relax_config", default_config),
        run_negf_config = config["step_configs"].get("run_negf_config", default_config),
        run_overlap_config = config["step_configs"].get("run_overlap_config", default_config),
        upload_python_packages=upload_python_packages
    )

    return Step(
        "negf",
        template=negf_op,
        parameters=parameters,
        artifacts=artifacts
    )


def get_systems_from_data(data, data_prefix=None):
    data = [data] if isinstance(data, str) else data
    assert isinstance(data, list)
    if data_prefix is not None:
        data = [os.path.join(data_prefix, ii) for ii in data]
    return data


def make_negf_op(
        relax_style,
        overlap_style,
        build_supercell_step_config,
        prep_relax_config,
        prep_negf_config,
        prep_overlap_config,
        run_relax_config,
        run_negf_config,
        run_overlap_config,
        upload_python_packages):
    if relax_style in relax_styles.keys():
        prep_run_relax_op = PrepRunRelax(
            "prep-run-relax",
            relax_styles[relax_style]["prep"],
            relax_styles[relax_style]["run"],
            prep_config=prep_relax_config,
            run_config=run_relax_config,
            upload_python_packages=upload_python_packages
        )
    else:
        raise RuntimeError(f"unknown relax_style {relax_style}")

    if overlap_style is not None:
        if overlap_style in overlap_styles.keys():
            prep_run_overlap_op = PrepRunOverlap(
                "prep-run-overlap",
                overlap_styles[overlap_style]["prep"],
                overlap_styles[overlap_style]["run"],
                prep_config=prep_overlap_config,
                run_config=run_overlap_config,
                upload_python_packages=upload_python_packages,
            )
        else:
            raise RuntimeError(f"unknown overlap_style {overlap_style}")
    else:
        prep_run_overlap_op = None

    prep_run_negf_op = PrepRunNegf(
        "prep-run-negf",
        prep_config=prep_negf_config,
        run_config=run_negf_config,
        upload_python_packages=upload_python_packages,
    )

    return NEGFSteps(
        "negf",
        build_supercell_op=BuildSupercell,
        prep_run_relax_op=prep_run_relax_op,
        prep_run_overlap_op=prep_run_overlap_op,
        prep_run_negf_op=prep_run_negf_op,
        build_supercell_step_config=build_supercell_step_config,
        upload_python_packages=upload_python_packages
    )


class NEGFSteps(Steps):
    """
    A class to represent the NEGF operation Steps.
    """

    def __init__(
            self,
            name: str,
            build_supercell_op: Type[OP],
            prep_run_relax_op: Type[OP],
            prep_run_overlap_op: Optional[OP],
            prep_run_negf_op: Type[OP],
            build_supercell_step_config: dict,
            upload_python_packages: Optional[List[os.PathLike]] = None
    ):
        self._input_parameters = {
            "negf_input_config": InputParameter(),
            "relax_input_config": InputParameter(),
            "overlap_input_config": InputParameter(),
            "task_config": InputParameter(),
            "relax_config": InputParameter(),
            "inputs_config": InputParameter(),
            "negf_config": InputParameter(),
            "overlap_config": InputParameter()
        }
        self._input_artifacts = {
            "init_confs": InputArtifact(),
            "deepmd_model": InputArtifact(),
            "deeptb_model": InputArtifact()
        }
        if prep_run_overlap_op is not None:
            self._input_artifacts["pp_orb"] = InputArtifact()

        self._output_parameters = {}
        self._output_artifacts = {
            "relaxed_systems": OutputArtifact(),
            "negf_out": OutputArtifact()
        }
        super().__init__(
            name=name,
            inputs=Inputs(
                parameters=self._input_parameters,
                artifacts=self._input_artifacts
            ),
            outputs=Outputs(
                parameters=self._output_parameters,
                artifacts=self._output_artifacts
            )
        )
        self = _negf(
            self,
            name,
            build_supercell_op=build_supercell_op,
            build_supercell_step_config=build_supercell_step_config,
            prep_run_relax_op=prep_run_relax_op,
            prep_run_negf_op=prep_run_negf_op,
            upload_python_packages=upload_python_packages
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


def _negf(
        steps,
        name: str,
        build_supercell_op: Type[OP],
        build_supercell_step_config: dict,
        prep_run_relax_op,
        prep_run_negf_op,
        upload_python_packages: Optional[List[os.PathLike]] = None
):
    """
    Creates the steps for the DataGen operation.

    Args:

    Returns:
        The steps object with the added steps.
    """
    build_supercell_step_config = deepcopy(build_supercell_step_config)
    build_supercell_template_config = build_supercell_step_config.pop("template_config")
    build_supercell_executor = init_executor(build_supercell_step_config.pop("executor"))

    build_supercell = Step(
        name + "-pert-gen",
        template=PythonOPTemplate(
            build_supercell_op,
            python_packages=upload_python_packages,
            **build_supercell_template_config
        ),
        parameters={"negf_config": steps.inputs.parameters["negf_config"]},
        artifacts={"init_confs": steps.inputs.artifacts["init_confs"]},
        key="build-supercell",
        executor=build_supercell_executor,
        **build_supercell_step_config
    )
    steps.add(build_supercell)

    prep_run_relax = Step(
        name=name + "-prep-run-relax",
        template=prep_run_relax_op,
        parameters={
            "relax_config": steps.inputs.parameters["relax_config"],
            "system_infos": build_supercell.outputs.parameters["system_infos"],
            "inputs_config": steps.inputs.parameters["inputs_config"]
        },
        artifacts={
            "stacked_systems": build_supercell.outputs.artifacts["stacked_systems"],
            "deepmd_model": steps.inputs.artifacts["deepmd_model"]
        },
        key="prep-run-relax"
    )
    steps.add(prep_run_relax)

    prep_run_negf = Step(
        name=name + "-prep-run-negf",
        template=prep_run_negf_op,
        parameters={
            "negf_input_config": steps.inputs.parameters["negf_input_config"],
            "task_infos": prep_run_relax.outputs.parameters["task_infos"],
            "task_config": steps.inputs.parameters["task_config"]
        },
        artifacts={
            "relaxed_systems": prep_run_relax.outputs.artifacts["relaxed_systems"],
            "deeptb_model": steps.inputs.artifacts["deeptb_model"]
        },
        key="prep-run-negf"
    )
    steps.add(prep_run_negf)

    """    prep_run_fp = Step(
        name=name + "-prep-run-fp",
        template=prep_run_fp_op,
        parameters={
            "block_id": "init",
            "fp_config": steps.inputs.parameters["fp_config"],
            "type_map": steps.inputs.parameters["type_map"],
        },
        artifacts={
            "confs": pert_gen.outputs.artifacts["confs"]
        },
        key="--".join(["init", "prep-run-fp"]),
    )
    steps.add(prep_run_fp)

    steps.outputs.artifacts["multi_systems"]._from = collect_data.outputs.artifacts[
        "multi_systems"
    ]"""

    return steps