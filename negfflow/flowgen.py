import os
from pathlib import Path
from typing import Union, Dict
from dflow import Workflow
from negfflow.args import normalize
import dflow
from negfflow.flow.negf import make_negf_step


class FlowGen:
    def __init__(
        self,
        config: Dict,
        debug: bool = False,
        download_path: Union[Path, str] = Path("../"),
    ):
        self._download_path = download_path
        if debug is True:
            os.environ["DFLOW_DEBUG"] = "1"
            dflow.config["mode"] = "debug"
        elif os.environ.get("DFLOW_DEBUG"):
            del os.environ["DFLOW_DEBUG"]
            dflow.config["mode"] = "default"
        self._config = normalize(config)
        print("dflow mode: %s" % dflow.config["mode"])
        self.workflow = Workflow(name=self._config["name"])
        self._wf_type = config["task"].get("type")
        if self._wf_type == "negf":
            self.workflow.add(make_negf_step(self._config))

    @property
    def wf_type(self):
        return self._wf_type

    @property
    def download_path(self):
        if isinstance(self._download_path, str):
            return Path(self._download_path)
        else:
            return self._download_path

    def submit(self):
        self.workflow.submit()


