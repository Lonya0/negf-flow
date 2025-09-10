from negfflow.op.relax.prep_lammps import PrepLammps
from negfflow.op.relax.run_lammps import RunLammps

relax_styles = {
    "lammps": {
        "prep": PrepLammps,
        "run": RunLammps
    }
}