from omegaconf import DictConfig
from pathlib import Path
from typing import Tuple, Dict, List

# ANSI escape codes for colors
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"


def validate_config(cfg: DictConfig):
    all_validation_results = []
    print(f"{BOLD}{CYAN}+++++ Validating Config{RESET}")

    if cfg.config_validation.validate_benchmark:
        benchmark_res, benchmark_valid = check_benchmark_config(cfg.benchmark)
        all_validation_results.append(benchmark_valid)
        print_benchmark_report(benchmark_res, benchmark_valid)
    else:
        print(f"{YELLOW}WARNING: Validation of benchmark config disabeled!{RESET}")

    if cfg.config_validation.validate_solver:
        solver_res, solver_valid = check_solver_config(cfg.solver)
        all_validation_results.append(solver_valid)
        print_solver_report(solver_res, solver_valid)
    else:
        print(f"{YELLOW}WARNING: Validation of solver config disabeled!{RESET}")

    # Check if format of solver and format of benchmark match
    if cfg.solver.format not in cfg.benchmark.format:
        all_validation_results.append(False)
        print(
            f"{RED}✘ Solver and benchmark format do not match ({cfg.solver.format}{RESET}"
        )
    else:
        print(
            f"{GREEN}✔ Solver and benchmark format match ({cfg.solver.format}){RESET}"
        )

    return all(all_validation_results)


def check_solver_config(config) -> Tuple[dict, bool]:
    # Things for solver to check:
    # - config.path exists
    return None, True


def print_solver_report(solver_validation_results: dict, is_valid: bool):
    print(f"{RED} SOLVER-REPORT{RESET}")


def check_benchmark_config(config: Dict) -> Tuple[Dict[str, bool], bool]:
    """
    Checks if the specified directory exists, verifies if files with specified
    extensions exist for all formats, and checks if the count of `format` files matches
    the count of `query_arg_format` files for each format, and if the number of instances
    for each format is equal.

    Args:
        config (dict): Configuration dictionary containing the fields:
                       - `path` (str): Directory path.
                       - `format` (list of str): List of file extensions for primary files.
                       - `query_arg_format` (str): File extension for query argument files.

    Returns:
        dict: Dictionary with the results of each check.
        bool: True if all checks are passed, False otherwise.
    """
    results = {
        "directory_exists": False,
        "all_formats_present": False,
        "matching_file_count": False,
        "equal_instance_count": False,
    }

    directory = Path(config["path"])
    formats = config["format"]  # Now a list of formats
    query_arg_ext = config.get("query_arg_format")

    # Check if directory exists
    if directory.is_dir():
        results["directory_exists"] = True

    # Initialize flags
    all_formats_present = True
    matching_file_count = True
    equal_instance_count = True

    # Collect file counts for each format
    format_file_counts = {}

    # Check each format in the list
    for format_ext in formats:
        format_files = list(directory.rglob(f"*.{format_ext}"))
        format_file_counts[format_ext] = len(format_files)

        # Check if files with the specified format exist
        if not format_files:
            all_formats_present = False

        # Check count of files matches the count of query_arg_format files if provided
        if query_arg_ext is not None:
            query_arg_files = list(directory.rglob(f"*.{query_arg_ext}"))
            if len(format_files) != len(query_arg_files):
                matching_file_count = False

    # Check if all formats have an equal number of instances
    if len(set(format_file_counts.values())) == 1:
        equal_instance_count = True
    else:
        equal_instance_count = False

    # Update results
    results["all_formats_present"] = all_formats_present
    results["matching_file_count"] = matching_file_count
    results["equal_instance_count"] = equal_instance_count
    results["file_counts"] = format_file_counts

    # Determine if all checks pass
    valid = all(results.values())
    return results, valid


def print_benchmark_report(benchmark_validation_results: dict, is_valid: bool):
    """
    Prints a detailed benchmark validation report based on the validation results.

    Args:
        benchmark_validation_results (dict): The results dictionary from the benchmark validation.
        is_valid (bool): Indicates if the benchmark configuration is valid.
    """

    print(benchmark_validation_results)
    if is_valid:
        print(f"{GREEN}✔ Benchmark configuration is valid.{RESET}")
    else:
        print(f"{RED}✘ Benchmark configuration is not valid.{RESET}")

    # Detailed report of each check
    if benchmark_validation_results["directory_exists"]:
        print(f"    {GREEN}✔ Directory exists{RESET}")
    else:
        print(f"    {RED}✘ Directory does not exist{RESET}")

    if benchmark_validation_results["all_formats_present"]:
        print(f"    {GREEN}✔ Files with specified format exist{RESET}")
    else:
        print(f"    {RED}✘ No files with specified format found{RESET}")

    if benchmark_validation_results["matching_file_count"]:
        print(
            f"    {GREEN}✔ File count matches for specified formats({benchmark_validation_results['file_counts']}){RESET}"
        )
    else:
        print(f"    {RED}✘ File count does not match for specified formats{RESET}")
