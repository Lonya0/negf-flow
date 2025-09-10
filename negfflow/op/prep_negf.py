import os
from typing import List
from dflow.python import OP, OPIO, OPIOSign, BigParameter

class PrepNegf(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "negf_input_config": dict,
            "task_infos": List[dict],
            "task_config": dict
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "task_names": BigParameter(List[str]),
            "modified_negf_input_configs": BigParameter(List[dict])
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        task_infos = op_in["task_infos"]
        negf_input_config = op_in["negf_input_config"]
        use_external_overlap = op_in["task_config"]["use_external_overlap"]

        task_names = []
        modified_negf_input_configs = []
        output_dir = "tasks"

        for task_info in task_infos:
            system_info = task_info["system_info"]

            os.makedirs(output_dir, exist_ok=True)

            modified_negf_input_config = negf_input_config.copy()
            modified_negf_input_config['task_options']['stru_options']['device']['id'] = f"{system_info['atom_index'][0]}-{system_info['atom_index'][1]}"
            modified_negf_input_config['task_options']['stru_options']['lead_L']['id'] = f"0-{system_info['atom_index'][0]}"
            modified_negf_input_config['task_options']['stru_options']['lead_R']['id'] = f"{system_info['atom_index'][1]}-{system_info['atom_index'][2]}"

            modified_negf_input_config['structure'] = "relaxed.vasp"

            task_name = f"negf_{task_info['conf_name']}_{task_info['temp']}K_{task_info['pres']}bar"

            task_names.append(task_name)
            modified_negf_input_configs.append(modified_negf_input_config)
            print(f"创建任务: {task_name}")

        op_out = OPIO({
            "modified_negf_input_configs": modified_negf_input_configs,
            "task_names": task_names
        })

        return op_out

    def __init__(self):
        return
