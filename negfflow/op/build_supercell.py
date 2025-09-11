import os.path
from typing import List
from pathlib import Path
from dflow.python import OP, OPIO, OPIOSign, Artifact, BigParameter
from ase.io import read, write
from ase import Atoms
import numpy as np

class BuildSupercell(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "init_confs": Artifact(List[Path]),
            "negf_config": dict
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "stacked_systems": Artifact(List[Path]),
            "system_infos": BigParameter(List[dict])
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        negf_config = op_in['negf_config']
        assert negf_config['supercell']['lead_L'] == negf_config['supercell']['lead_R'], "lead_L should be equal to lead_R for symmetric leads."
        assert negf_config['supercell']['lead_L'] % 2 == 0, "lead should be in even number as double principal layers."

        def stack(init_system, _output_file, negf_config):
            # di = direction index, sm = super cell matrix
            if negf_config['direction'] == 'x': (di, sm) = (0, np.array([1, 0, 0]))
            elif negf_config['direction'] == 'y': (di, sm) = (1, np.array([0, 1, 0]))
            elif negf_config['direction'] == 'z': (di, sm) = (2, np.array([0, 0, 1]))
            else: raise TypeError(f"direction {negf_config['direction']} is not legal!")

            # sort in direction
            directional_coords = init_system.positions[:, di]
            sorted_indices = np.argsort(directional_coords)
            sorted_system = Atoms(
                symbols=init_system.symbols[sorted_indices],
                positions=init_system.positions[sorted_indices],
                cell=init_system.cell,
                pbc=init_system.pbc
            )

            # build supercell
            repeat = sum(negf_config['supercell'].values())
            supercell = sorted_system.repeat((1, 1, 1) + (repeat - 1) * sm)

            # switch 1-2 principal layers
            pos = supercell.get_positions()
            n_cell = int(negf_config['supercell']['lead_L'] / 2) # how many unit cells in one principal layer
            atom_number_of_layer = len(init_system) * n_cell
            cell_c = sorted_system.cell[di, di]
            new_pos = np.vstack([pos[:atom_number_of_layer] + sm * cell_c * n_cell,
                                 pos[atom_number_of_layer:2 * atom_number_of_layer] - sm * cell_c * n_cell, 
                                 pos[2 * atom_number_of_layer:]])
            supercell.set_positions(new_pos)

            write(_output_file, supercell, format='vasp')
            return (negf_config['supercell']['lead_L'], 
                    negf_config['supercell']['device'], 
                    negf_config['supercell']['lead_R'])

        out_systems = []
        system_infos = []

        for conf in op_in["init_confs"]:
            with open(conf, "r", encoding="utf-8") as f:
                system = read(f)
            output_file = 'stacked_' + os.path.basename(conf)
            ll, dd, rr = stack(system, output_file, negf_config)
            out_systems.append(Path.cwd() / output_file)
            system_infos.append({'atom_number': len(system),
                                 'atom_index': list(np.array([ll, ll + dd, ll + dd + rr]) * 
                                                    system.get_number_of_atoms())})

        op_out = OPIO({
            "stacked_systems": out_systems,
            "system_infos": system_infos
        })

        return op_out
