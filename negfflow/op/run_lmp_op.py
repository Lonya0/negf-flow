from dflow.python import OP, OPIO, OPIOSign, Artifact
from pathlib import Path

class run_lmp_op(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "msg": str,
            "num": int,
            "foo": Artifact(Path),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "msg": str,
            "bar": Artifact(Path),
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        with open(op_in["foo"], "r") as f:
            content = f.read()
        with open("bar.txt", "w") as f:
            f.write(content * op_in["num"])

        op_out = OPIO({
            "msg": op_in["msg"] * op_in["num"],
            "bar": Path("bar.txt"),
        })
        return op_out