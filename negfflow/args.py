import textwrap
from typing import (
    List
)

import dargs
from dargs import (
    Argument,
    Variant,
)


def make_link(content, ref_key):
    raw_anchor = dargs.dargs.RAW_ANCHOR
    return (
        f"`{content} <{ref_key}_>`_" if not raw_anchor else f"`{content} <#{ref_key}>`_"
    )


def relax_arg():
    doc_relax = "The configuration for exploration"

    def variant_relax():
        doc = "The type of the relaxation"
        doc_lammps = "The relaxation by LAMMPS simulations"

        return Variant(
            "type",
            [
                Argument("lammps", dict, lammps_field(), doc=doc_lammps, alias=['lmp'])
            ],
            doc=doc,
        )

    def lammps_field():
        doc_config = "Configuration file path for lammps relaxation"
        doc_ensemble = "Maximum number of iterations per stage"
        doc_dt = "dt"
        doc_nsteps = "nsteps"
        doc_temps = "temps"
        doc_press = "press"
        doc_run_config = "Configuration for running lammps"

        def run_config_field():
            doc_cmd = "The command of abacus"
            return [
                Argument("command", str, optional=True, default="lmp", doc=doc_cmd),
            ]

        return [
            Argument("config", str, optional=True, default=None, doc=doc_config),
            Argument("ensemble", str, optional=False, doc=doc_ensemble),
            Argument("dt", float, optional=True, default=True, doc=doc_dt),
            Argument("nsteps", int, optional=True, default=True, doc=doc_nsteps),
            Argument("temps", [int, List[int]], optional=False, doc=doc_temps),
            Argument("press", [int, List[int]], optional=False, doc=doc_press),
            Argument("run_config", dict, run_config_field(), optional=False, doc=doc_run_config)
        ]

    return [
        Argument(
            "relax", dict, [], [variant_relax()], optional=False, doc=doc_relax, alias=["relaxation"],
        )
    ]


def inputs_arg():
    """
    The input parameters and artifacts of dptb-negf workflow
    """
    doc_inputs = "config of input"

    def inputs_field():
        doc_init_confs_uri = "Uri of initial systems"
        doc_init_confs_prefix = "The prefix of initial systems"
        doc_init_confs_paths = "Path of initial systems"
        doc_deepmd_model_uri = "deepmd_model_uri"
        doc_deepmd_model_path = "deepmd_model_path"
        doc_deepmd_model_type_map = "type map of deepmd model, you can get this by 'dp show {model} type-map'"
        doc_deeptb_model_uri = "deeptb_model_uri"
        doc_deeptb_model_path = "deeptb_model_path"
        return [
            Argument("init_confs_uri", str, optional=True, default=None, doc=doc_init_confs_uri),
            Argument("init_confs_prefix", str, optional=True, default='.', doc=doc_init_confs_prefix),
            Argument("init_confs_paths", List[str], optional=False, doc=doc_init_confs_paths),
            Argument("deepmd_model_uri", str, optional=True, default=None, doc=doc_deepmd_model_uri, alias=['dpmd_model_uri']),
            Argument("deepmd_model_path", str, optional=True, default=None, doc=doc_deepmd_model_path, alias=['dpmd_model_path']),
            Argument("deepmd_model_type_map", list, optional=True, default=None, doc=doc_deepmd_model_type_map, alias=['dpmd_model_type_map']),
            Argument("deeptb_model_uri", str, optional=True, default=None, doc=doc_deeptb_model_uri, alias=['dptb_model_uri']),
            Argument("deeptb_model_path", str, optional=True, default=None, doc=doc_deeptb_model_path, alias=['dptb_model_path'])
            ]

    return [
        Argument("inputs", dict, inputs_field(), optional=False, doc=doc_inputs)
    ]



def wf_args():
    doc_name = "The workflow name, 'negfflow' for default"
    doc_step_configs = "Configurations for executing dflow steps"

    """def dflow_conf_args():
        doc_dflow_config = "The configuration passed to dflow"
        doc_dflow_s3_config = "The S3 configuration passed to dflow"

        return [
            Argument(
                "dflow_config", dict, optional=True, default=None, doc=doc_dflow_config
            ),
            Argument(
                "dflow_s3_config",
                dict,
                optional=True,
                default=None,
                doc=doc_dflow_s3_config,
            ),
        ]

    def bohrium_conf_args():
        doc_username = "The username of the Bohrium platform"
        doc_password = "The password of the Bohrium platform"
        doc_project_id = "The project ID of the Bohrium platform"
        doc_host = (
            "The host name of the Bohrium platform. Will overwrite `dflow_config['host']`"
        )
        doc_k8s_api_server = "The k8s server of the Bohrium platform. Will overwrite `dflow_config['k8s_api_server']`"
        doc_repo_key = "The repo key of the Bohrium platform. Will overwrite `dflow_s3_config['repo_key']`"
        doc_storage_client = "The storage client of the Bohrium platform. Will overwrite `dflow_s3_config['storage_client']`"

        return [
            Argument("username", str, optional=False, doc=doc_username),
            Argument("password", str, optional=True, doc=doc_password),
            Argument("project_id", int, optional=False, doc=doc_project_id),
            Argument("ticket", str, optional=True),
            Argument(
                "host",
                str,
                optional=True,
                default="https://workflows.deepmodeling.com",
                doc=doc_host,
            ),
            Argument(
                "k8s_api_server",
                str,
                optional=True,
                default="https://workflows.deepmodeling.com",
                doc=doc_k8s_api_server,
            ),
            Argument(
                "repo_key", str, optional=True, default="oss-bohrium", doc=doc_repo_key
            ),
            Argument(
                "storage_client",
                str,
                optional=True,
                default="dflow.plugins.bohrium.TiefblueClient",
                doc=doc_storage_client,
            ),
        ]"""

    def template_conf_args():
        doc_image = "The image to run the step."
        doc_timeout = "The time limit of the op. Unit is second."
        doc_retry_on_transient_error = (
            "The number of retry times if a TransientError is raised."
        )
        doc_timeout_as_transient_error = "Treat the timeout as TransientError."
        doc_envs = "The environmental variables."
        return [
            Argument("image", str, optional=True, default=None, doc=doc_image),
            Argument("timeout", int, optional=True, default=None, doc=doc_timeout),
            Argument(
                "retry_on_transient_error",
                int,
                optional=True,
                default=None,
                doc=doc_retry_on_transient_error,
            ),
            Argument(
                "timeout_as_transient_error",
                bool,
                optional=True,
                default=False,
                doc=doc_timeout_as_transient_error,
            ),
            Argument("envs", dict, optional=True, default=None, doc=doc_envs),
        ]

    def template_slice_conf_args():
        doc_group_size = "The number of tasks running on a single node. It is efficient for a large number of short tasks."
        doc_pool_size = "The number of tasks running at the same time on one node."
        return [
            Argument("group_size", int, optional=True, default=None, doc=doc_group_size),
            Argument("pool_size", int, optional=True, default=None, doc=doc_pool_size),
        ]

    def step_config_args():
        doc_template = "The configs passed to the PythonOPTemplate."
        doc_template_slice = "The configs passed to the Slices."
        doc_executor = "The executor of the step."
        doc_continue_on_failed = "If continue the the step is failed (FatalError, TransientError, A certain number of retrial is reached...)."
        doc_continue_on_num_success = "Only in the sliced op case. Continue the workflow if a certain number of the sliced jobs are successful."
        doc_continue_on_success_ratio = "Only in the sliced op case. Continue the workflow if a certain ratio of the sliced jobs are successful."
        doc_parallelism = "The parallelism for the step"

        def variant_executor():
            doc = f"The type of the executor."

            def dispatcher_args():
                """free style dispatcher args"""
                return []

            return Variant(
                "type",
                [
                    Argument("dispatcher", dict, dispatcher_args()),
                ],
                doc=doc,
            )

        return [
            Argument(
                "template_config",
                dict,
                template_conf_args(),
                optional=True,
                default={"image": None},
                doc=doc_template,
            ),
            Argument(
                "template_slice_config",
                dict,
                template_slice_conf_args(),
                optional=True,
                doc=doc_template_slice,
            ),
            Argument(
                "continue_on_failed",
                bool,
                optional=True,
                default=False,
                doc=doc_continue_on_failed,
            ),
            Argument(
                "continue_on_num_success",
                int,
                optional=True,
                default=None,
                doc=doc_continue_on_num_success,
            ),
            Argument(
                "continue_on_success_ratio",
                float,
                optional=True,
                default=None,
                doc=doc_continue_on_success_ratio,
            ),
            Argument("parallelism", int, optional=True, default=None, doc=doc_parallelism),
            Argument(
                "executor",
                dict,
                [],
                [variant_executor()],
                optional=True,
                default=None,
                doc=doc_executor,
            ),
        ]

    def default_step_config():
        base = Argument("base", dict, step_config_args())
        data = base.normalize_value({}, trim_pattern="_*")
        # not possible to strictly check dispatcher arguments, dirty hack!
        base.check_value(data, strict=False)
        return data

    def default_step_config_args():
        doc_default_step_config = "The default step configuration."

        return [
            Argument("default_step_config", dict, step_config_args(), optional=True, default=default_step_config(),
                     doc=doc_default_step_config)
        ]

    def step_configs_field():
        doc_prep_relax_config = "Configuration for relax"
        doc_prep_negf_config = "Configuration for negf"
        doc_prep_overlap_config = "Configuration for overlap"
        doc_run_relax_config = "Configuration for relax"
        doc_run_negf_config = "Configuration for negf"
        doc_run_overlap_config = "Configuration for overlap"

        return [
            Argument("prep_relax_config", dict, step_config_args(), optional=True, default=default_step_config(),
                     doc=doc_prep_relax_config),
            Argument("prep_negf_config", dict, step_config_args(), optional=True, default=default_step_config(),
                     doc=doc_prep_negf_config, alias=['prep_dptb_config']),
            Argument("prep_overlap_config", dict, step_config_args(), optional=True, default=default_step_config(),
                     doc=doc_prep_overlap_config),
            Argument("run_relax_config", dict, step_config_args(), optional=True, default=default_step_config(),
                     doc=doc_run_relax_config),
            Argument("run_negf_config", dict, step_config_args(), optional=True, default=default_step_config(),
                     doc=doc_run_negf_config, alias=['run_dptb_config']),
            Argument("run_overlap_config", dict, step_config_args(), optional=True, default=default_step_config(),
                     doc=doc_run_overlap_config)
        ]

    return (
        [Argument("name", str, optional=True, default="negfflow", doc=doc_name)]
        + default_step_config_args()
        + [Argument("step_configs", dict, step_configs_field(), optional=True, default={}, doc=doc_step_configs)]
    )


def task_arg():
    doc_task = "Task type, `finetune` or `dist`"

    def variant_task():
        return Variant(
            "type",
            [
                Argument("negf", dict, task_negf(), alias=[])
            ],
        )

    def task_negf():
        doc_use_external_overlap = "Using external generated overlap"
        return [
            Argument("use_external_overlap", bool, optional=True, default=False, doc=doc_use_external_overlap)
        ]

    return [
        Argument("task", dict, [], [variant_task()], optional=False, doc=doc_task)
    ]


def negf_arg():
    doc_negf = "negf construction options"

    def supercell_field():
        doc_lead_L = "lead_L"
        doc_device = "device"
        doc_lead_R = "lead_R"
        return [
            Argument("lead_L", int, optional=False, doc=doc_lead_L),
            Argument("device", int, optional=False, doc=doc_device),
            Argument("lead_R", int, optional=False, doc=doc_lead_R)
        ]

    def negf_field():
        doc_config = "config"
        doc_supercell = "supercell"
        doc_direction = "direction"
        return [
            Argument("config", str, optional=False, doc=doc_config),
            Argument("supercell", dict, supercell_field(), optional=False, doc=doc_supercell),
            Argument("direction", str, optional=False, doc=doc_direction)
        ]

    return [
        Argument("negf", int, negf_field(), optional=False, doc=doc_negf)
    ]


def overlap_arg():
    doc_overlap = "The configuration for overlap"

    def variant_overlap():
        doc = "The type of the overlap source"
        doc_abacus = "Generate overlap by abacus"

        return Variant(
            "type",
            [
                Argument("abacus", dict, abacus_field(), doc=doc_abacus)
            ],
            doc=doc,
        )

    def abacus_field():
        doc_run_config = "Configuration for running abacus"
        doc_inputs_config = "input config"

        def run_config_field():
            """from dpgen2"""
            doc_cmd = "The command of abacus"
            return [
                Argument("command", str, optional=True, default="abacus", doc=doc_cmd),
            ]

        def inputs_config_field():
            """from dpgen2"""
            doc_input_file = "A template INPUT file."
            doc_pp_files = (
                "The pseudopotential files for the elements. "
                'For example: {"H": "/path/to/H.upf", "O": "/path/to/O.upf"}.'
            )
            doc_element_mass = (
                "Specify the mass of some elements. "
                'For example: {"H": 1.0079, "O": 15.9994}.'
            )
            doc_kpt_file = "The KPT file, by default None."
            doc_orb_files = (
                "The numerical orbital fiels for the elements, "
                "by default None. "
                'For example: {"H": "/path/to/H.orb", "O": "/path/to/O.orb"}.'
            )
            doc_deepks_descriptor = "The deepks descriptor file, by default None."
            doc_deepks_model = "The deepks model file, by default None."
            return [
                Argument("input_file", str, optional=False, doc=doc_input_file),
                Argument("pp_files", dict, optional=False, doc=doc_pp_files),
                Argument(
                    "element_mass", dict, optional=True, default=None, doc=doc_element_mass
                ),
                Argument("kpt_file", str, optional=True, default=None, doc=doc_kpt_file),
                Argument("orb_files", dict, optional=True, default=None, doc=doc_orb_files),
                Argument(
                    "deepks_descriptor",
                    str,
                    optional=True,
                    default=None,
                    doc=doc_deepks_descriptor,
                ),
                Argument(
                    "deepks_model", str, optional=True, default=None, doc=doc_deepks_model
                ),
            ]

        return [
            Argument("run_config", dict, run_config_field(), optional=False, doc=doc_run_config),
            Argument("inputs_config", str, inputs_config_field(), optional=False, doc=doc_inputs_config)
        ]

    return [
        Argument(
            "overlap", dict, [], [variant_overlap()], optional=True, doc=doc_overlap
        )
    ]


def submit_args():
    return (
        wf_args()
        + task_arg()
        + inputs_arg()
        + negf_arg()
        + relax_arg()
        + overlap_arg()
    )


def normalize(data):
    defs = submit_args()
    base = Argument("base", dict, defs)
    data = base.normalize_value(data, trim_pattern="_*")
    # not possible to strictly check arguments, dirty hack!
    base.check_value(data, strict=False)
    return data


def gen_doc(*, make_anchor=True, make_link=True, **kwargs):
    if make_link:
        make_anchor = True
    sca = submit_args()
    base = Argument("submit", dict, sca)
    ptr = []
    ptr.append(base.gen_doc(make_anchor=make_anchor, make_link=make_link, **kwargs))
    key_words = []
    for ii in "\n\n".join(ptr).split("\n"):
        if "argument path" in ii:
            key_words.append(ii.split(":")[1].replace("`", "").strip())
    return "\n\n".join(ptr)
