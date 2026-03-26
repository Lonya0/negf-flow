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

        def stack(init_system, _output_file, _negf_config):
            # di = direction index, sm = super cell matrix
            if _negf_config['direction'] == 'x': (di, sm) = (0, np.array([1, 0, 0]))
            elif _negf_config['direction'] == 'y': (di, sm) = (1, np.array([0, 1, 0]))
            elif _negf_config['direction'] == 'z': (di, sm) = (2, np.array([0, 0, 1]))
            else: raise TypeError(f"direction {_negf_config['direction']} is not legal!")

            # sort in direction
            directional_coords = init_system.positions[:, di]
            sorted_indices = np.argsort(directional_coords)
            sorted_system = Atoms(
                symbols=init_system.symbols[sorted_indices],
                positions=init_system.positions[sorted_indices],
                cell=init_system.cell,
                pbc=init_system.pbc
            )

            # unified length
            if 'auto_scale_length' in _negf_config.keys():
                if _negf_config['direction'] == 'x':
                    stru_length = sorted_system.cell.lengths()[0]
                    sm = np.array([1, 0, 0])
                elif _negf_config['direction'] == 'y':
                    stru_length = sorted_system.cell.lengths()[1]
                    sm = np.array([0, 1, 0])
                elif _negf_config['direction'] == 'z':
                    stru_length = sorted_system.cell.lengths()[2]
                    sm = np.array([0, 0, 1])
                else:
                    raise TypeError(f"direction {_negf_config['direction']} is not legal!")
                mult = int(_negf_config['auto_scale_length'] // stru_length)
                sorted_system = sorted_system.repeat((1, 1, 1) + (mult - 1) * sm)

                # sort again
                directional_coords = sorted_system.positions[:, di]
                sorted_indices = np.argsort(directional_coords)
                sorted_system = Atoms(
                    symbols=sorted_system.symbols[sorted_indices],
                    positions=sorted_system.positions[sorted_indices],
                    cell=sorted_system.cell,
                    pbc=sorted_system.pbc
                )
            else:
                mult = 1

            # build supercell
            repeat = sum(_negf_config['supercell'].values())
            supercell = sorted_system.repeat((1, 1, 1) + (repeat - 1) * sm)

            # switch 1-2 principal layers
            pos = supercell.get_positions()
            n_cell = int(_negf_config['supercell']['lead_L'] / 2) # how many unit cells in one principal layer
            atom_number_of_layer = len(sorted_system) * n_cell
            cell_c = sorted_system.cell[di, di]
            new_pos = np.vstack([pos[:atom_number_of_layer] + sm * cell_c * n_cell,
                                 pos[atom_number_of_layer:2 * atom_number_of_layer] - sm * cell_c * n_cell, 
                                 pos[2 * atom_number_of_layer:]])
            supercell.set_positions(new_pos)

            write(_output_file, supercell, format='vasp')
            return (_negf_config['supercell']['lead_L'],
                    _negf_config['supercell']['device'],
                    _negf_config['supercell']['lead_R'],
                    mult)

        out_systems = []
        system_infos = []

        for conf in op_in["init_confs"]:
            with open(conf, "r", encoding="utf-8") as f:
                system = read(f)
            output_file = 'stacked_' + os.path.basename(conf)
            ll, dd, rr, _mult = stack(system, output_file, negf_config)
            out_systems.append(Path.cwd() / output_file)
            system_infos.append({'atom_number': len(system) * _mult,
                                 'atom_index': list(np.array([ll, ll + dd, ll + dd + rr]) * 
                                                    system.get_number_of_atoms() * _mult)})

        op_out = OPIO({
            "stacked_systems": out_systems,
            "system_infos": system_infos
        })

        return op_out
