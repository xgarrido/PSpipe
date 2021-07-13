import argparse
import collections
import logging
import os
from copy import deepcopy

import yaml
from simple_slurm import Slurm


def dict_merge(dct, merge_dct):
    dct = dct.copy()
    for k, v in merge_dct.items():
        dct[k] = (
            dict_merge(dct[k], v)
            if isinstance(dct.get(k), dict) and isinstance(v, collections.Mapping)
            else v
        )
    return dct


def main():
    parser = argparse.ArgumentParser(description="A python pipe for CMB analysis")
    parser.add_argument(
        "-d", "--debug", help="Enable debug level", default=False, action="store_true"
    )
    parser.add_argument(
        "-f",
        "--force",
        help="Overwrite previous output, if it exists",
        default=False,
        action="store_true",
    )
    parser.add_argument("-p", "--pipeline", help="Set pipeline file", required=True)
    parser.add_argument("-c", "--config", help="Set config file", required=True)
    parser.add_argument(
        "-v",
        "--var",
        help="Set variable to be overloaded in config file",
        action="append",
        default=list(),
    )

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    logging.debug("Reading pipeline file...")
    with open(args.pipeline, "r") as f:
        pipe_dict = yaml.safe_load(f)

    logging.debug("Reading configuration file...")
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    config_dict.update(
        {
            "debug": pipe_dict.get("debug", False) or args.debug,
            "force": pipe_dict.get("force", False) or args.force,
        }
    )
    for arg in args.var:
        keys, d = arg.split("=")
        logging.debug(f"Updating '{keys.replace('.', '/')}' value...")
        for key in reversed(keys.split(".")):
            d = {key: d}
        config_dict = dict_merge(config_dict, d)

    product_dir = config_dict.get("product_dir", ".")
    os.makedirs(product_dir, exist_ok=True)
    updated_yml = os.path.join(product_dir, "{}_updated{}".format(*os.path.splitext(args.config)))
    with open(updated_yml, "w") as f:
        yaml.dump(config_dict, f)

    slurm = pipe_dict.get("slurm", {})
    default_kwargs = dict(
        ntasks=slurm.get("ntasks", os.environ.get("SLURM_JOB_NUM_NODES", 1)),
        cpus_per_task=slurm.get("cpus_per_task", os.environ.get("SLURM_TASKS_PER_NODE", 64)),
    )
    for k, v in pipe_dict.get("pipeline", {}).items():
        slurm_kwargs = deepcopy(default_kwargs)
        slurm_kwargs.update(**v if v is not None else {})
        slurm = Slurm(**slurm_kwargs)
        script_file = f"{os.path.join(pipe_dict.get('script_base_dir', '.'), k)}.py"
        if not os.path.exists(script_file):
            raise ValueError(f"File {script_file} does not exist!")
        slurm.srun(f"python {script_file} {updated_yml}")


if __name__ == "__main__":
    main()
