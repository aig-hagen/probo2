import hydra
from omegaconf import DictConfig, OmegaConf
from functions import solver_interfaces
from utils import hydra_utils, config_validater
from hydra.core.utils import JobReturn
from typing import Any, List
from hydra.experimental.callback import Callback

from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from pathlib import Path

import os

import tqdm




def run_solver_dynamic_accaptance(cfg:DictConfig) -> None:
    pass
def run_solver_dynamic_enumeration(cfg:DictConfig) -> None:
    pass


def run_solver_static_accaptance(cfg:DictConfig) -> None:
    # - Basically the same as enumeration, but we need to create a lookup table for the query arguments
    # - create temp files for the look ups to save time: Check if present, load mapping, if not create
    # -


    pass

# Method to run static enumeration task such as SE or EE
def run_solver_static_enumeration(cfg: DictConfig) -> None:

    matching_format = hydra_utils.get_matching_format(
        cfg.solver.format, cfg.benchmark.format
    )

    if matching_format is None:
        print(
            f"No matching formats for {cfg.solver.format=} and {cfg.benchmark.format=}"
        )
        exit()

    instances = hydra_utils.get_files_with_extension(
        cfg.benchmark.path, matching_format
    )

    solver_interface_command = solver_interfaces.interfaces_dict[cfg.solver.interface](
        cfg
    )

    # Get the indicies of the options that change from solver call to solver call
    if cfg.solver.interface == "legacy":
        index_file_format_flag = solver_interface_command.index("-fo")
        solver_interface_command[index_file_format_flag + 1] = matching_format

    # Insert the file after '-f' flag
    # This is the only thing that changes from call to call for a solver
    index_option_to_change = solver_interface_command.index("-f") + 1

    desc = f"{cfg.solver.name}"

    hydra_config = HydraConfig.get()

    run_dir = hydra_config.runtime.output_dir
    output_root_dir = os.path.join(run_dir, hydra_config.output_subdir)

    result_file_path = os.path.join(output_root_dir,'result.csv')

    hydra_utils.write_result_file_to_index(result_file_path,index_file=cfg.result_index_file)

    for instance in tqdm.tqdm(instances, desc=desc):

        # Set instance
        solver_interface_command[index_option_to_change] = instance
        instance_name = Path(instance).stem
        current_output_file_path = os.path.join(output_root_dir,
                                                f"{instance_name}.out")

        result = hydra_utils.run_solver_with_timeout(
            solver_interface_command, cfg.timeout, current_output_file_path
        )

        # append additional info like solver, benchmark and general infos
        prefixed_solver_info = hydra_utils.add_prefix_to_dict_keys(cfg.solver,'solver_')
        prefixed_benchmark_info = hydra_utils.add_prefix_to_dict_keys(cfg.benchmark,'benchmark_')
        result.update(prefixed_solver_info)
        result.update(prefixed_benchmark_info)
        result.update({'experiment_name': cfg.name,'run': cfg.runs,'task': cfg.task,'timeout': cfg.timeout})

        # Write results to file
        hydra_utils.write_result_to_csv(data_dict=result,path=result_file_path)




@hydra.main(
    version_base=None, config_path="hydra_experiments_configs", config_name="config"
)
def run_experiment(cfg: DictConfig) -> None:
    # Generate custom directories
    os.makedirs(cfg.evaluation_output_dir, exist_ok=True)

    # Change the format string to a list
    if isinstance(cfg.benchmark.format, str):
        cfg.benchmark.format = [cfg.benchmark.format]  # Wrap single string in a list

    if cfg.config_validation.validate_config:
        config_valid = config_validater.validate_config(cfg)
        if not config_valid:
            print("Experiment aborted due to bad config.")
            print(
                "You can deactivate the config validater by setting validate_config: False in the config.yaml"
            )
            return
        print("Config is valid. Start running experiment.")
    # Run experiment based on task or config
    # Check if the cfg.task string contains the prefix SE or EE
    if cfg.task.startswith("SE") or cfg.task.startswith("EE"):
        run_solver_static_enumeration(cfg)
    # Check if the cfg.task string contains the prefix DC or DS
    elif cfg.task.startswith("DC") or cfg.task.startswith("DS"):
        run_solver_static_accaptance(cfg)



if __name__ == "__main__":
    run_experiment()
