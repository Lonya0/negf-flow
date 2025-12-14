import json
import os
from copy import deepcopy
from typing import Type, Optional, List

import dflow
import dpdata
from dflow import Step, Steps, InputParameter, InputArtifact, OutputArtifact, Inputs, Outputs, argo_sequence, argo_len, \
    OutputParameter
from dflow.python import OP, PythonOPTemplate, Slices
import fpop
import negfflow
from negfflow.op.build_supercell import BuildSupercell
from negfflow.utils.artifact import get_artifact_from_uri, upload_artifact_and_print_uri
from negfflow.utils.executor import init_executor


def make_press_band_step(config):
    default_config = config["default_step_config"]

    upload_python_packages = []
    if custom_packages := config.get("upload_python_packages"):
        upload_python_packages.extend(custom_packages)
    upload_python_packages.extend(list(dpdata.__path__))
    upload_python_packages.extend(list(dflow.__path__))
    upload_python_packages.extend(list(fpop.__path__))
    upload_python_packages.extend(list(negfflow.__path__))

    ## load parameters

    parameters = {
        "press_band_config": config['press_band'],
        "task_config": config['task'],
        "relax_config": config['relax'],
        "inputs_config": config['inputs']
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

    press_band_op = make_press_band_op(
        build_supercell_step_config=config["step_configs"].get("build_supercell_step_config", default_config),
        prep_press_config=config["step_configs"].get("prep_press_config", default_config),
        prep_dptb_config=config["step_configs"].get("prep_dptb_config", default_config),
        run_press_config=config["step_configs"].get("run_press_config", default_config),
        run_dptb_config=config["step_configs"].get("run_dptb_config", default_config),
        upload_python_packages=upload_python_packages
    )

    return Step(
        "press_band",
        template=press_band_op,
        parameters=parameters,
        artifacts=artifacts
    )

def get_systems_from_data(data, data_prefix=None):
    data = [data] if isinstance(data, str) else data
    assert isinstance(data, list)
    if data_prefix is not None:
        data = [os.path.join(data_prefix, ii) for ii in data]
    return data

import os.path
from typing import List
from pathlib import Path
from dflow.python import OP, OPIO, OPIOSign, Artifact, BigParameter
from ase.io import read, write
from ase.data import atomic_numbers, atomic_masses

class PrepPress(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "stacked_systems": Artifact(List[Path]),
            "press_config": dict,
            "system_infos": BigParameter(List[dict]),
            "inputs_config": dict
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "task_paths": Artifact(List[Path]),
            "task_names": BigParameter(List[str]),
            "task_infos": BigParameter(List[dict])
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        system_infos = op_in["system_infos"]
        press_config = op_in["press_config"]
        model_name = os.path.basename(op_in["inputs_config"]["deepmd_model_path"])
        confs = op_in["stacked_systems"]

        task_paths = []
        task_names = []
        task_infos = []
        output_dir = "tasks"

        for conf, system_info in zip(confs, system_infos):
            try:
                with open(conf, "r", encoding="utf-8") as f:
                    system = read(f)
            except Exception as e:
                print(f"读取VASP文件失败: {e}")
                continue

            os.makedirs(output_dir, exist_ok=True)

            """            fixed_atom_indices = (
                list(range(1, system_info['atom_index'][0] + 1)) +
                list(range(system_info['atom_index'][1] + 1, system_info['atom_index'][2] + 1))
            )"""


            if "deepmd_model_type_map" in op_in["inputs_config"] and op_in["inputs_config"]["deepmd_model_type_map"]:
                specorder = op_in["inputs_config"]["deepmd_model_type_map"]
            else:
                specorder = _build_specorder(system)

            for temp in press_config['temps']:
                for pres in press_config['press']:
                    task_info = {
                        "conf_name": os.path.basename(conf).replace('.vasp', ''),
                        "ensemble": press_config['ensemble'],
                        "temp": temp,
                        "pres": pres,
                        "system_info": system_info
                    }
                    task_name = f"lmp_relax_{task_info['conf_name']}_{task_info['temp']}K_{task_info['pres']}bar"
                    task_dir = os.path.join(output_dir, task_name)
                    os.makedirs(task_dir, exist_ok=True)

                    lammps_data_path = os.path.join(task_dir, "lammps.data")
                    write(lammps_data_path, system, format="lammps-data", specorder=specorder)

                    mass_lines = _mass_lines(specorder)
                    group_lines = _group_fixed_by_ids(None)
                    ensemble_block = _ensemble_block(press_config['ensemble'], temp, pres, press_config['dt'], press_config['nsteps'])

                    if press_config.get("nonaxial_press"):
                        press_block = """
# Indenter: rigid cylinder pressing along y-direction
variable x0 equal 0.5*lx
variable z0 equal 0.5*lz
variable y0 equal 0.9*ly

variable R equal 4.0
variable v_press equal -0.0002

fix ind all indent 50.0 cylinder y ${x0} ${z0} ${R} units box
fix move_ind none move linear 0.0 ${v_press} 0.0 units box

thermo_style custom step temp pe f_ind pxx pyy pzz
thermo 100
"""
                    else:
                        press_block = ""

                    in_lammps_path = os.path.join(task_dir, "in.lammps")
                    with open(in_lammps_path, "w", encoding="utf-8") as f:
                        f.write(f"""# Auto-generated
units           metal
atom_style      atomic
boundary        p p p

read_data       lammps.data

{mass_lines}

pair_style      deepmd {model_name}
pair_coeff      * * {' '.join(specorder)}

neighbor        2.0 bin
neigh_modify    every 10 delay 0 check yes

{group_lines}
fix             hold fixed setforce 0.0 0.0 0.0

{press_block}

{ensemble_block}

write_data      relaxed.data
        """)

                    task_paths.append(Path(task_dir))
                    task_names.append(task_name)
                    task_infos.append(task_info)
                    print(f"创建任务: {task_name}")

        op_out = OPIO({
            "task_paths": task_paths,
            "task_names": task_names,
            "task_infos": task_infos
        })

        return op_out

    def __init__(self):
        return

def _build_specorder(system):
    spec = []
    for s in system.get_chemical_symbols():
        if s not in spec:
            spec.append(s)
    return spec

def _mass_lines(specorder):
    lines = []
    for i, sym in enumerate(specorder, start=1):
        z = atomic_numbers[sym]
        m = float(atomic_masses[z])
        lines.append(f"mass {i} {m:.6f}  # {sym}")
    return "\n".join(lines)

def _group_fixed_by_ids(fixed_ids):
    if not fixed_ids:
        return "# no fixed atoms\ngroup mobile all"
    ids = [str(i) for i in sorted(set(fixed_ids))]
    lines = [f"group fixed id {' '.join(ids)}",
             "group mobile subtract all fixed"]
    return "\n".join(lines)

def _ensemble_block(ensemble, T, P, dt, nsteps):
    if ensemble.lower() == "min":
        return (
            "thermo          100\n"
            "min_style       cg\n"
            f"minimize        1e-6 1e-8 1000 {nsteps}"
        )
    if ensemble.lower() == "nvt":
        return (
            f"velocity        mobile create {T} 12345 mom yes rot yes dist gaussian\n"
            f"fix             1 mobile nvt temp {T} {T} 0.1\n"
            "thermo          100\n"
            f"timestep        {dt}\n"
            f"run             {nsteps}"
        )
    if ensemble.lower() == "npt":
        return (
            f"velocity        mobile create {T} 12345 mom yes rot yes dist gaussian\n"
            f"fix             1 mobile npt temp {T} {T} 0.1 iso {P:.6f} {P:.6f} 1.0 dilate mobile\n"
            "thermo          100\n"
            f"timestep        {dt}\n"
            f"run             {nsteps}"
        )
    if ensemble.lower() == "nve":
        return (
            f"velocity        mobile create {T} 12345 mom yes rot yes dist gaussian\n"
            "fix             1 mobile nve\n"
            "thermo          100\n"
            f"timestep        {dt}\n"
            f"run             {nsteps}"
        )
    raise ValueError("ensemble 必须是 'min'/'nvt'/'npt'/'nve' 之一")

import logging
from pathlib import Path
import os.path
from dflow.python import OP, OPIO, OPIOSign, Artifact, BigParameter
from dflow.python.python_op_template import TransientError
from ase.io import read, write
from negfflow.utils.set_directory import set_directory
from negfflow.utils.run_command import run_command
from negfflow.utils.safe_symlink import safe_symlink

class RunPress(OP):

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "task_path": Artifact(Path),
            "task_name": BigParameter(str),
            "deepmd_model": Artifact(Path),
            "press_config": dict
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "log": Artifact(Path),
            "extra_outputs": Artifact(Path, optional=True),
            "relaxed_system": Artifact(Path),
            "task_name": BigParameter(str)
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        task_name = op_in["task_name"]
        task_path = op_in["task_path"]
        deepmd_model = op_in["deepmd_model"]
        command = op_in["press_config"]["run_config"]["command"]
        work_dir = Path(task_name)

        with set_directory(work_dir):
            safe_symlink("in.lammps", task_path / "in.lammps")
            safe_symlink("lammps.data", task_path / "lammps.data")
            safe_symlink(os.path.basename(deepmd_model), deepmd_model)
            command = " ".join([command, "-i", 'in.lammps', "-log", "log.lammps"])
            ret, out, err = run_command(command, shell=True)
            if ret != 0:
                logging.error(
                    "".join(
                        (
                            "lmp failed\n",
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
                raise TransientError("lmp failed")

            relaxed_system = read("relaxed.data", format='lammps-data')
            write("relaxed.vasp", relaxed_system, vasp5=True)

        op_out = OPIO({
            "log": Path(work_dir) / "log.lammps",
            "extra_outputs": None,
            "relaxed_system": Path(work_dir) / "relaxed.vasp",
            "task_name": task_name
        })

        return op_out

    def __init__(self):
        return


class PrepRunPress(Steps):
    def __init__(self, name, prep_config, run_config, upload_python_packages):
        self._input_parameters = {
            "press_band_input_config": InputParameter(),
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
            "press_band_results": OutputArtifact(),
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

        prep_op = PrepPress()
        run_op = RunPress()

        self._keys = ["prep-press", "run-press"]
        self.step_keys = {}
        self.step_keys["prep-press"] = "prep-press"
        self.step_keys["run-press"] = "run-press-{{item}}"

        self = _prep_run_press(
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

def _prep_run_press(
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

    prep_press = Step(
        "prep-press",
        template=PythonOPTemplate(
            prep_op,
            python_packages=upload_python_packages,
            **prep_template_config,
        ),
        parameters={
            "task_infos": prep_run_steps.inputs.parameters["task_infos"],
            "task_config": prep_run_steps.inputs.parameters["task_config"]
        },
        artifacts={},
        key=step_keys["prep-press"],
        executor=prep_executor,
        **prep_config,
    )
    prep_run_steps.add(prep_press)

    run_press = Step(
        "run-press",
        template=PythonOPTemplate(
            run_op,
            slices=Slices(
                "int('{{item}}')",
                input_parameter=["task_name", "modified_press_band_input_config"],
                input_artifact=["relaxed_system"],
                output_artifact=["log", "press_band_result", "extra_outputs"],
                **template_slice_config,
            ),
            python_packages=upload_python_packages,
            **run_template_config,
        ),
        parameters={
            "task_name": prep_press.outputs.parameters["task_names"],
            "modified_press_band_input_config": prep_press.outputs.parameters["modified_press_input_configs"],
        },
        artifacts={
            "relaxed_system": prep_run_steps.inputs.artifacts["relaxed_systems"],
            "deeptb_model": prep_run_steps.inputs.artifacts["deeptb_model"]
        },
        with_sequence=argo_sequence(
            argo_len(prep_press.outputs.parameters["task_names"])
        ),
        key=step_keys["run-press"],
        executor=run_executor,
        **run_config
    )
    prep_run_steps.add(run_press)

    prep_run_steps.outputs.parameters["task_names"].value_from_parameter = prep_press.outputs.parameters["task_names"]
    prep_run_steps.outputs.artifacts["logs"]._from = run_press.outputs.artifacts[
        "log"
    ]
    prep_run_steps.outputs.artifacts["press_band_results"]._from = run_press.outputs.artifacts[
        "press_band_result"
    ]
    prep_run_steps.outputs.artifacts["extra_outputs"]._from = run_press.outputs.artifacts[
        "extra_outputs"
    ]

    return prep_run_steps


def make_press_band_op(
        build_supercell_step_config,
        prep_press_config,
        prep_dptb_config,
        run_press_config,
        run_dptb_config,
        upload_python_packages):
    prep_run_press_op = PrepRunPress(
            "prep-run-press",
            prep_config=prep_press_config,
            run_config=run_press_config,
            upload_python_packages=upload_python_packages
        )

    prep_run_dptb_op = PrepRunPress(
        "prep-run-dptb",
        prep_config=prep_dptb_config,
        run_config=run_dptb_config,
        upload_python_packages=upload_python_packages,
    )

    return PressBandSteps(
        "press_band",
        build_supercell_op=BuildSupercell,
        prep_run_press_op=prep_run_press_op,
        prep_run_dptb_op=prep_run_dptb_op,
        build_supercell_step_config=build_supercell_step_config,
        upload_python_packages=upload_python_packages
    )


class PressBandSteps(Steps):
    """
    A class to represent the PRESS_BAND operation Steps.
    """

    def __init__(
            self,
            name: str,
            build_supercell_op: Type[OP],
            prep_run_press_op: Type[OP],
            prep_run_dptb_op: Type[OP],
            build_supercell_step_config: dict,
            upload_python_packages: Optional[List[os.PathLike]] = None
    ):
        self._input_parameters = {
            "press_band_input_config": InputParameter(),
            "relax_input_config": InputParameter(),
            "overlap_input_config": InputParameter(),
            "task_config": InputParameter(),
            "relax_config": InputParameter(),
            "inputs_config": InputParameter(),
            "press_band_config": InputParameter(),
            "overlap_config": InputParameter()
        }
        self._input_artifacts = {
            "init_confs": InputArtifact(),
            "deepmd_model": InputArtifact(),
            "deeptb_model": InputArtifact()
        }

        self._output_parameters = {}
        self._output_artifacts = {
            "relaxed_systems": OutputArtifact(),
            "press_band_results": OutputArtifact(),
            "extra_outputs": OutputArtifact(optional=True)
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
        self = _press_band(
            self,
            name,
            build_supercell_op=build_supercell_op,
            build_supercell_step_config=build_supercell_step_config,
            prep_run_press_op=prep_run_press_op,
            prep_run_dptb_op=prep_run_dptb_op,
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


def _press_band(
        steps,
        name: str,
        build_supercell_op: Type[OP],
        build_supercell_step_config: dict,
        prep_run_press_op,
        prep_run_dptb_op,
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
        parameters={"press_band_config": steps.inputs.parameters["press_band_config"]},
        artifacts={"init_confs": steps.inputs.artifacts["init_confs"]},
        key="build-supercell",
        executor=build_supercell_executor,
        **build_supercell_step_config
    )
    steps.add(build_supercell)

    prep_run_press = Step(
        name=name + "-prep-run-press",
        template=prep_run_press_op,
        parameters={
            "press_config": steps.inputs.parameters["press_config"],
            "system_infos": build_supercell.outputs.parameters["system_infos"],
            "inputs_config": steps.inputs.parameters["inputs_config"]
        },
        artifacts={
            "stacked_systems": build_supercell.outputs.artifacts["stacked_systems"],
            "deepmd_model": steps.inputs.artifacts["deepmd_model"]
        },
        key="prep-run-press"
    )
    steps.add(prep_run_press)

    prep_run_dptb = Step(
        name=name + "-prep-run-dptb",
        template=prep_run_dptb_op,
        parameters={
            "dptb_input_config": steps.inputs.parameters["dptb_input_config"],
            "task_infos": prep_run_press.outputs.parameters["task_infos"],
            "task_config": steps.inputs.parameters["task_config"]
        },
        artifacts={
            "relaxed_systems": prep_run_press.outputs.artifacts["relaxed_systems"],
            "deeptb_model": steps.inputs.artifacts["deeptb_model"]
        },
        key="prep-run-dptb"
    )
    steps.add(prep_run_dptb)

    steps.outputs.artifacts["relaxed_systems"]._from = prep_run_press.outputs.artifacts["relaxed_systems"]
    steps.outputs.artifacts["dptb_results"]._from = prep_run_dptb.outputs.artifacts[
        "dptb_results"
    ]
    steps.outputs.artifacts["extra_outputs"]._from = prep_run_dptb.outputs.artifacts[
        "extra_outputs"
    ]

    return steps
