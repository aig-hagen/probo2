import hydra
from omegaconf import DictConfig, OmegaConf
from functions import solver_interfaces
from utils import hydra_utils, config_validater

from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from pathlib import Path

import os

import tqdm


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
        cfg.benchmark.path, cfg.solver.format
    )

    solver_interface_command = solver_interfaces.interfaces_dict[cfg.solver.interface](
        cfg
    )

    # Get the indicies of the options that change from solver call to solver call
    if cfg.solver.interface == "legacy":
        solver_interface_command[-1] = matching_format

    # Insert the file after '-f' flag
    index_option_to_change = solver_interface_command.index("-f") + 1

    desc = f"{cfg.solver.name}"

    # Access Hydra's runtime config (includes runtime paths like run dir)
    hydra_config = HydraConfig.get()

    # print(hydra_config.output_subdir)

    # # Hydra's run directory (where the outputs and logs are stored)
    # output_dir = run_dir  # In most cases, output_dir is the same as run_dir by default

    # # Log file path (example, assuming logs are stored in the output directory)
    # log_file_path = f"{output_dir}/logs/my_logfile.log"

    # print(f"Run directory: {run_dir}")
    # print(f"Output directory: {output_dir}")
    # print(f"Log file path: {log_file_path}")

    # Optional: If you need the original working directory (before Hydra changed it)

    run_dir = hydra_config.runtime.output_dir
    output_root_dir = os.path.join(run_dir, hydra_config.output_subdir)

    for instance in tqdm.tqdm(instances, desc=desc):

        # Set instance
        solver_interface_command[index_option_to_change] = instance
        instance_name = Path(instance).stem
        current_output_file_path = os.path.join(output_root_dir,
                                                f"{instance_name}.out")

        hydra_utils.run_solver_with_timeout(
            solver_interface_command, cfg.timeout, current_output_file_path
        )

        # hydra_utils.run_solver_with_timeout(command=solver_interface_command,)

        # result = solver_handler.run_solver(solver, task, config.timeout, instance, format, additional_arguments_lookup,dynamic_files_lookup,output_file_dir=solver_output_dir,repetition=rep,solver_options=solver_options)
        # result.update(benchmark_info)
        # result['repetition'] = rep
        # result['tag'] = config.name
        # write_result(result,result_path,config.result_format)
        # if rep == 1:
        #     Status.increment_instances_counter(config,task,solver['id'])


@hydra.main(
    version_base=None, config_path="hydra_experiments_configs", config_name="config"
)
def my_app(cfg: DictConfig) -> None:

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

    run_solver_static_enumeration(cfg)


if __name__ == "__main__":
    my_app()
