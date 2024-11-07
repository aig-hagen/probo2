

import hydra
from omegaconf import DictConfig, OmegaConf
from functions import solver_interfaces


import subprocess
import time
import os
import signal
import tqdm

from pathlib import Path
from collections import OrderedDict

def get_files_with_extension(directory: str, extension: str):
    # Use pathlib to get files with the given extension
    return [str(file) for file in Path(directory).rglob(f'*.{extension}') if file.is_file()]

def run_binary_with_timeout(command, timeout, output_file):
    start_time = time.time()  # Record start time
    try:
        # Start the process with a new process group
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid  # Start in a new process group (Linux/macOS)
        )
        
        # Wait for the process to complete or timeout
        stdout, stderr = process.communicate(timeout=timeout)
        end_time = time.time()
        time_taken = end_time - start_time
        status = "Success" if process.returncode == 0 else "Failed"
        
        # Write results to file
        with open(output_file, "w") as file:
            file.write(stdout)
            # file.write(f"Command: {' '.join(command)}\n")
            # file.write(f"Status: {status}\n")
            # file.write(f"Time taken: {time_taken:.2f} seconds\n")
            # file.write("Output:\n")
            # file.write(stdout)
            # file.write("\nErrors:\n")
            # file.write(stderr)

    except subprocess.TimeoutExpired:
        # Timeout case: kill the process group
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)  # Terminate the process group
        
        # Write timeout status to file
        with open(output_file, "w") as file:
            file.write('TIMEOUT')
            

# Example usage
#run_binary_with_timeout(["your_binary", "arg1", "arg2"], timeout=5, output_file="output.txt")


def need_additional_arguments(task: str):
    if 'DC-' in task or 'DS-' in task:
        return True
    else:
        return False


def prepare_instances(cfg: DictConfig):
    if cfg.solver.format == 'i23':
        return get_files_with_extension(cfg.benchmark.path,cfg.benchmark.format)
    else:
        return get_files_with_extension(cfg.benchmark.path,cfg.solver.format)

def run_solver(cfg: DictConfig) -> None:
    instances = prepare_instances(cfg)
    solver_interface_command = solver_interfaces.interfaces_dict[cfg.solver.interface](cfg)

  
        
    
    desc = f'{cfg.solver.name}'

    for instance in tqdm.tqdm(instances,desc=desc):

        continue
        
        # result = solver_handler.run_solver(solver, task, config.timeout, instance, format, additional_arguments_lookup,dynamic_files_lookup,output_file_dir=solver_output_dir,repetition=rep,solver_options=solver_options)
        # result.update(benchmark_info)
        # result['repetition'] = rep
        # result['tag'] = config.name
        # write_result(result,result_path,config.result_format)
        # if rep == 1:
        #     Status.increment_instances_counter(config,task,solver['id'])



@hydra.main(version_base=None, config_path="hydra_experiments_config", config_name="config")
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    run_solver(cfg)

if __name__ == "__main__":
    my_app()