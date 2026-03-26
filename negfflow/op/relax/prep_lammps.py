import os.path
from typing import List
from pathlib import Path
from dflow.python import OP, OPIO, OPIOSign, Artifact, BigParameter
from ase.io import read, write
from ase.data import atomic_numbers, atomic_masses


class PrepLammps(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "stacked_systems": Artifact(List[Path]),
            "relax_config": dict,
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
        relax_config = op_in["relax_config"]
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

            fixed_atom_indices = (
                list(range(1, system_info['atom_index'][0] + 1)) +
                list(range(system_info['atom_index'][1] + 1, system_info['atom_index'][2] + 1))
            )

            if "device_end_fixed_radius" in relax_config:
                radius = relax_config["device_end_fixed_radius"]

                cell = system.get_cell()
                cell_z = cell[2][2]

                atom_number = system_info['atom_number']
                supercell_multiplier = system_info['atom_index'][2] / atom_number
                cell_length = cell_z / supercell_multiplier

                a0, a1 = system_info['atom_index'][0], system_info['atom_index'][1]

                z0_limit = (a0 / atom_number) * cell_length + radius
                z1_limit = (a1 / atom_number) * cell_length - radius

                positions = system.get_positions()
                for idx in range(a0, a1):
                    atom_id = idx + 1
                    z = positions[idx][2]

                    if z < z0_limit or z > z1_limit:
                        fixed_atom_indices.append(atom_id)

            if "deepmd_model_type_map" in op_in["inputs_config"] and op_in["inputs_config"]["deepmd_model_type_map"]:
                specorder = op_in["inputs_config"]["deepmd_model_type_map"]
            else:
                specorder = _build_specorder(system)

            for temp in relax_config['temps']:
                for pres in relax_config['press']:
                    task_info = {
                        "conf_name": os.path.basename(conf).replace('.vasp', ''),
                        "ensemble": relax_config['ensemble'],
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
                    group_lines = _group_fixed_by_ids(fixed_atom_indices)
                    ensemble_block = _ensemble_block(relax_config['ensemble'],
                                                     temp,
                                                     pres,
                                                     relax_config['dt'],
                                                     relax_config['nsteps'],
                                                     relax_config['additional']['temp_0'],
                                                     relax_config['additional']['nstep_heating'],
                                                     relax_config['additional']['nstep_equil'],
                                                     relax_config['additional']['n_out'])

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
fix freeze fixed spring/self 1e8

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

"""
def _group_fixed_by_ids(fixed_ids):
    if not fixed_ids:
        return "# no fixed atoms\ngroup mobile all"
    ids = [str(i) for i in sorted(set(fixed_ids))]
    chunks = [ids[i:i+15] for i in range(0, len(ids), 15)]
    lines, tmps = [], []
    for k, ch in enumerate(chunks, 1):
        g = f"gfix{k}"
        tmps.append(g)
        lines.append(f"group {g} id {' '.join(ch)}")
    if len(tmps) == 1:
        lines.append(f"group fixed union {tmps[0]}")
    else:
        lines.append(f"group fixed union {' '.join(tmps)}")
    for g in tmps:
        lines.append(f"group {g} delete")
    lines.append("group mobile subtract all fixed")
    return "\n".join(lines)"""

def _ensemble_block(ensemble,
                    T,
                    P,
                    dt,
                    nsteps,
                    temp_0 = 50,
                    nstep_heating = 10000,
                    nstep_equil = 10000,
                    n_out = 10):
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
    if ensemble.lower() == "smart":
        return (
            "\n".join(["# -----------------------------",
            "# 0. 能量最小化",
            "min_style       cg",
            "minimize        1e-10 1e-10 10000 10000",
            "reset_timestep  0",
            "# -----------------------------",
            "# 1. 初始速度（低温启动）",
            f"velocity mobile create {temp_0} 12345 mom yes rot yes dist gaussian",
            "# -----------------------------",
            "# 2. 缓慢升温（50K → 目标温度）",
            f"variable T equal {T}",
            f"fix heat mobile nvt temp {temp_0} {T} 0.1",
            "thermo 100",
            "timestep 0.001",
            f"run {nstep_heating}     # 50 ps 升温",
            "unfix heat",
            "reset_timestep 0",
            "# -----------------------------",
            "# 3. 恒温弛豫（达到热平衡）",
            "fix equil mobile nvt temp ${T} ${T} 0.1",
            "dump eq all atom 500 equil.xyz",
            f"run {nstep_equil}     # 50 ps 平衡",
            "unfix equil",
            "undump eq",
            "reset_timestep 0",
            "# -----------------------------",
            "# 4. 采样阶段（关键：系综平均）",
            "fix sample mobile nvt temp ${T} ${T} 0.1",
            "# 每1000步输出一个结构（约1 ps间隔）",
            "dump traj all custom 1000 traj.xyz id type x y z",
            'dump_modify traj format line "%d %d %.16f %.16f %.16f"',
            f"run {1000 * (n_out - 1)}    # 100 ps → 得到约100个构型",
            "unfix sample",
            "undump traj",
            "write_data final.data"])
        )
    raise ValueError("ensemble 必须是 'min'/'nvt'/'npt'/'nve' 之一")
