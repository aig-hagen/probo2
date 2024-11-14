import csv
import json
from csv import DictWriter, DictReader
from src.handler import config_handler
import os
import pandas as pd

from src.handler import solver_handler, benchmark_handler
from src.utils import definitions
from tqdm import tqdm
import shutil
import tabulate
from src.utils import Status
from click import echo

from src.functions import plot, statistics, score, printing, table_export
from src.functions import (
    validation,
    plot_validation,
    validation_table_export,
    print_validation,
)
from src.functions import parametric_significance, parametric_post_hoc
from src.functions import non_parametric_significance, non_parametric_post_hoc
from src.functions import (
    plot_significance,
    post_hoc_table_export,
    plot_post_hoc,
    print_significance,
)
import src.functions.register as register
from functools import reduce
import colorama
from datetime import datetime
from enum import Enum


class ExperimentStatus(Enum):
    ABORTED = "Aborted"
    RUNNING = "Running"
    FINISHED = "Finished"
    NONE = "None"  # Represents the initial status


def run_pipeline(cfg: config_handler.Config):
    run_experiment(cfg)

    if cfg.copy_raws:
        echo("Copying raw files...", nl=False)
        copy_raws(cfg)
        echo("done!")

    result_df = load_results_via_name(cfg.name)

    saved_file_paths = []
    if cfg.plot is not None:
        saved_plots = plot.create_plots(result_df, cfg)

    to_merge = []
    others = []
    if cfg.statistics is not None:

        if cfg.statistics == "all" or "all" in cfg.statistics:
            cfg.statistics = register.stat_dict.keys()
        stats_results = []
        print("========== STATISTICS ==========")
        for stat in cfg.statistics:
            _res = register.stat_dict[stat](result_df)
            stats_results.append(_res)

        for res in stats_results:
            if res[1]:
                to_merge.append(res[0])
            else:
                others.append(res[0])
    if cfg.score is not None:
        score_results = []
        if cfg.score == "all" or "all" in cfg.score:
            cfg.score = register.score_functions_dict.keys()
        for s in cfg.score:
            _res = register.score_functions_dict[s](result_df)
            score_results.append(_res)
        for res in score_results:
            if res[1]:
                to_merge.append(res[0])
            else:
                others.append(res[0])
    if len(to_merge) > 0:
        df_merged = reduce(
            lambda left, right: pd.merge(left, right, how="inner"), to_merge
        )
        register.print_functions_dict[cfg.printing](
            df_merged, ["tag", "task", "benchmark_name"]
        )
        if cfg.table_export is not None:
            if cfg.table_export == "all" or "all" in cfg.table_export:
                cfg.table_export = register.table_export_functions_dict.keys()
            for format in cfg.table_export:
                register.table_export_functions_dict[format](
                    df_merged, cfg, ["tag", "task", "benchmark_name"]
                )

    if cfg.validation["mode"]:
        validation_results = validation.validate(result_df, cfg)
        print_validation.print_results(validation_results)
        if cfg.validation["plot"]:
            plot_validation.create_plots(validation_results["pairwise"], cfg)
        if "pairwise" in cfg.validation["mode"]:
            if cfg.validation["table_export"]:
                if (
                    cfg.validation["table_export"] == "all"
                    or "all" in cfg.validation["table_export"]
                ):
                    cfg.validation["table_export"] = (
                        register.validation_table_export_functions_dict.keys()
                    )
                for f in cfg.validation["table_export"]:
                    register.validation_table_export_functions_dict[f](
                        validation_results["pairwise"], cfg
                    )

    test_results = {}
    post_hoc_results = {}

    if cfg.significance["parametric_test"]:
        test_results.update(parametric_significance.test(result_df, cfg))
    if cfg.significance["non_parametric_test"]:
        test_results.update(non_parametric_significance.test(result_df, cfg))
    if cfg.significance["parametric_post_hoc"]:
        post_hoc_results.update(parametric_post_hoc.test(result_df, cfg))
    if cfg.significance["non_parametric_post_hoc"]:
        post_hoc_results.update(non_parametric_post_hoc.test(result_df, cfg))

    if test_results:
        print("========== Significance Analysis Summary ==========")
        for test in test_results.keys():
            print_significance.print_results(test_results[test], test)

    if post_hoc_results:
        print("========== Post-hoc Analysis Summary ==========")
        for test in post_hoc_results.keys():
            print_significance.print_results_post_hoc(post_hoc_results[test], test)

        if cfg.significance["plot"]:
            for post_hoc_test in post_hoc_results.keys():
                plot_post_hoc.create_plots(
                    post_hoc_results[post_hoc_test], cfg, post_hoc_test
                )
        if cfg.significance["table_export"]:
            if (
                cfg.significance["table_export"] == "all"
                or "all" in cfg.significance["table_export"]
            ):
                cfg.significance["table_export"] = (
                    register.post_hoc_table_export_functions_dict.keys()
                )
            for post_hoc_test in post_hoc_results.keys():
                for f in cfg.significance["table_export"]:
                    register.post_hoc_table_export_functions_dict[f](
                        post_hoc_results[post_hoc_test], cfg, post_hoc_test
                    )

    if cfg.archive is not None:
        echo("Creating archives...", nl=False)
        for _format in cfg.archive:
            register.archive_functions_dict[_format](cfg.save_to)
        echo("done!")


def need_additional_arguments(task: str):
    if "DC-" in task or "DS-" in task:
        return True
    else:
        return False


def need_dynamic_arguments(task: str):
    if task.endswith("-D"):
        return True
    else:
        return False


def need_dynamic_files_lookup(task):
    return False


def get_accepted_format(solver_formats, benchmark_formats):
    _shared = list(set.intersection(set(solver_formats), set(benchmark_formats)))

    if _shared:
        return _shared[0]
    else:
        return None


def _format_benchmark_info(benchmark_info):
    formatted = {}
    for key in benchmark_info.keys():
        formatted[f"benchmark_{key}"] = benchmark_info[key]
    return formatted


def _append_result_directoy_suffix(config: config_handler.Config):
    suffix = 1
    while True:
        experiment_name = f"{config.name}_{suffix}"
        result_file_directory = os.path.join(
            definitions.RESULT_DIRECTORY, experiment_name
        )
        if not os.path.exists(result_file_directory):
            config.name = experiment_name
            return result_file_directory
        else:
            suffix += 1


def init_result_path(config: config_handler.Config, result_file_directory):

    if os.path.exists(result_file_directory):
        result_file_directory = _append_result_directoy_suffix(config)

    os.makedirs(result_file_directory, exist_ok=True)
    result_file_path = os.path.join(
        result_file_directory, f"raw.{config.result_format}"
    )
    if os.path.exists(result_file_path):
        suffix = len(os.listdir(result_file_directory))
        result_file_path = os.path.join(
            result_file_directory, f"raw_{suffix}.{config.result_format}"
        )

    return result_file_path


def load_results_via_name(name):

    experiment_index = pd.read_csv(definitions.EXPERIMENT_INDEX)

    if name in experiment_index.name.values:
        cfg_path = experiment_index[experiment_index.name.values == name][
            "config_path"
        ].iloc[0]
        cfg = config_handler.load_config_yaml(cfg_path, as_obj=True)
        return load_experiments_results(cfg)
    else:
        print(f"Experiment with name {name} does not exist.")
        exit()


def _write(format, file, content):
    if format == "json":
        file.seek(0)
        json.dump(content, file)
    elif format == "csv":

        header = content.keys()
        dictwriter_object = DictWriter(file, fieldnames=header)
        dictwriter_object.writerow(content)


def _write_initial(format, file, content):
    if format == "json":
        json.dump(content, file)
    elif format == "csv":
        header = content.keys()
        dictwriter_object = DictWriter(file, fieldnames=header)
        dictwriter_object.writeheader()
        dictwriter_object.writerow(content)


def _read(format, file):
    if format == "json":
        return json.load(file)
    elif format == "csv":
        return [row for row in DictReader(file)]


def write_result(result, result_path, result_format):
    """
    Writes a new experiment entry to the experiment index file with an automatically generated ID and current timestamp.

    This function appends a new record to an existing CSV file (or creates one if it does not exist) that
    stores experiment data. It automatically generates a unique ID for each new experiment, appends the
    current timestamp, and sets the initial status to 'None'.

    Parameters:
    - config (config_handler.Config): An object containing configuration details of the experiment. This
      object must have attributes 'name', 'raw_results_path', and 'yaml_file_name'.
    - result_directory_path (str): The directory path where the experiment results are stored.

    Returns:
    - int: The ID assigned to the new experiment entry.

    Notes:
    - This function assumes there is a globally accessible 'definitions.EXPERIMENT_INDEX' which contains the path
      to the index file.
    - The 'ExperimentStatus' enum is used to set the initial status of the experiment to 'None'.
    - The ID for the new entry is determined by counting the existing entries in the file using the `get_next_id`
      function.
    - The CSV file structure includes the columns: id, name, raw_path, config_path, timestamp and status.
    - If the CSV file is empty or not found, the function will write a header first and start IDs from 1.

    Example usage:
    ```python
    from config_handler import Config

    # Create a configuration instance
    config = Config(name="Test Experiment", raw_results_path="/path/to/raw/data", yaml_file_name="config.yaml")
    result_directory_path = "/path/to/results"

    # Write experiment data and get the new experiment ID
    new_experiment_id = write_experiment_index(config, result_directory_path)
    ```
    """

    if not os.path.exists(result_path):
        with open(result_path, "w") as result_file:
            _write_initial(result_format, result_file, result)

    else:
        with open(result_path, "a+") as result_file:
            # result_data = _read(result_format, result_file)
            # result_data.append(result)
            _write(result_format, result_file, result)


def write_experiment_index(config: config_handler.Config, result_directory_path):
    header = ["id", "name", "raw_path", "config_path", "timestamp", "status"]

    next_id = get_next_id()
    # Current timestamp
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    initial_status = ExperimentStatus.NONE.value
    # Write to file
    with open(definitions.EXPERIMENT_INDEX, "a", newline="") as fd:
        writer = csv.writer(fd)
        if next_id == 1 or next_id == 0:  # File was empty or not found, write header
            writer.writerow(header)
            next_id = 1
        # Write the new experiment entry
        writer.writerow(
            [
                next_id,
                config.name,
                config.raw_results_path,
                os.path.join(result_directory_path, config.yaml_file_name),
                current_time,
                initial_status,
            ]
        )
    return next_id


def get_next_id():
    """
    Determines the next unique ID for an experiment by counting the existing entries in the index file.

    This function reads the experiment index file specified in the 'definitions.EXPERIMENT_INDEX' path,
    counts the number of entries including the header, and returns an integer representing the next
    available ID. The first row of the CSV file is assumed to be a header, so the count starts from the
    second row onward.

    Returns:
    - int: The next available ID based on the number of entries in the file.

    Raises:
    - FileNotFoundError: If the index file does not exist, indicating that no entries have been recorded
      yet and the function will return 1, signifying that the first ID should be 1.

    Notes:
    - This function assumes that the file is well-formed (i.e., contains a header and follows a consistent
      row structure). It also assumes that no IDs are skipped in the file, meaning the next ID should
      always be `number of rows` since the header is not an experiment entry.

    Example usage:
    ```python
    next_experiment_id = get_next_id()
    ```
    """
    # Determine the next ID based on the number of existing entries
    try:
        with open(definitions.EXPERIMENT_INDEX, "r") as file:
            reader = csv.reader(file)
            existing_rows = list(reader)
            next_id = len(existing_rows)  # Assumes the first row is the header

        return next_id
    except FileNotFoundError:
        next_id = 1
        return next_id  # File does not exist yet, start with ID 1


def set_experiment_status(
    index_file_path: str, experiment_id: int, new_status: ExperimentStatus
):
    """
    Updates the status of a specific experiment in an index file.

    This function reads an index file that lists experiments, searches for the experiment with
    the given ID, and updates its status if the experiment is found and the new status is valid.
    The entire file is read into memory, modified, and then written back to ensure the update is
    saved.

    Parameters:
    - index_file_path (str): The path to the CSV file containing the experiment records.
    - experiment_id (int): The unique identifier of the experiment whose status is to be updated.
    - new_status (ExperimentStatus): The new status to set for the experiment, which must be one
      of the values defined in the ExperimentStatus enum.

    Notes:
    - The CSV file should have a header, and the experiment records should be structured as follows:
      ID, name, raw_path, config_path, timestamp, status.
    - The function checks if the new_status is valid based on the ExperimentStatus enum values before
      updating. If the new_status is not valid or the specified ID does not exist, no changes will be made.
    - The status of the experiment is updated in-place if found. All rows, including unchanged ones,
      are written back to the file.

    Raises:
    - FileNotFoundError: If the index_file_path does not exist.
    - IOError: If there is an error reading from or writing to the file.
    - ValueError: If the new_status is not a valid ExperimentStatus.

    Example usage:
    ```python
    set_experiment_status('/path/to/experiment/index.csv', 1, ExperimentStatus.RUNNING)
    ```
    """
    updated_rows = []
    status_updated = False
    valid_status = [
        status.value for status in ExperimentStatus
    ]  # List of valid statuses

    with open(index_file_path, "r", newline="") as fd:
        reader = csv.reader(fd)
        for row in reader:
            if row[0] == str(experiment_id) and new_status.value in valid_status:
                row[5] = new_status.value
                status_updated = True
            updated_rows.append(row)

    if status_updated:
        with open(index_file_path, "w", newline="") as fd:
            writer = csv.writer(fd)
            writer.writerows(updated_rows)


def load_experiments_results(config: config_handler.Config) -> pd.DataFrame:
    if os.path.exists(config.raw_results_path):
        if config.result_format == "json":
            return pd.read_json(config.raw_results_path)
        elif config.result_format == "csv":
            return pd.read_csv(config.raw_results_path)
    else:
        print(f"Results for experiment {config.name} not found!")


def exclude_task(config: config_handler.Config):
    to_exclude = []
    for current_task in config.task:
        for task in config.exclude_task:
            if task in current_task:
                to_exclude.append(current_task)

    config.task = [task for task in config.task if task not in to_exclude]


def set_supported_tasks(solver_list, config: config_handler.Config):
    supported_tasks = [set(solver["tasks"]) for solver in solver_list]
    supported_set = set.union(*supported_tasks)
    config.task = list(supported_set)


def run_experiment(config: config_handler.Config):

    solver_list = solver_handler.load_solver(config.solver)
    benchmark_list = benchmark_handler.load_benchmark(config.benchmark)
    if config.task == "supported":
        set_supported_tasks(solver_list, config)
    if config.exclude_task is not None:
        exclude_task(config)

    additional_arguments_lookup = None
    dynamic_files_lookup = None

    # if config.solver_arguments:
    #  config_handler.create_solver_argument_grid(config.solver_arguments, solver_list)

    experiment_result_directory = os.path.join(
        definitions.RESULT_DIRECTORY, config.name
    )
    result_path = init_result_path(config, experiment_result_directory)
    config.raw_results_path = result_path
    if config.save_to is None:
        config.save_to = os.path.join(os.getcwd(), config.name)
    else:
        config.save_to = os.path.join(config.save_to, config.name)
    cfg_experiment_result_directory = os.path.join(
        definitions.RESULT_DIRECTORY, config.name
    )
    status_file_path = os.path.join(cfg_experiment_result_directory, "status.json")
    config.status_file_path = status_file_path
    Status.init_status_file(config)
    config.dump(cfg_experiment_result_directory)
    experiment_id = write_experiment_index(config, cfg_experiment_result_directory)

    print(colorama.Fore.GREEN + "========== Experiment Summary ==========")
    config.print()
    print(colorama.Fore.GREEN + "========== RUNNING EXPERIMENT ==========")
    print(
        f"Start running experiment {colorama.Fore.YELLOW + config.name + colorama.Style.RESET_ALL} with ID: {colorama.Fore.YELLOW  + str(experiment_id) + colorama.Style.RESET_ALL}"
    )
    set_experiment_status(
        definitions.EXPERIMENT_INDEX, experiment_id, ExperimentStatus.RUNNING
    )
    for task in config.task:
        print(f"+TASK: {task}")
        for benchmark in benchmark_list:
            benchmark_info = _format_benchmark_info(benchmark)
            print(f" +BENCHMARK: {benchmark_info['benchmark_name']}")
            if (
                need_additional_arguments(task)
                and len(benchmark_info["benchmark_ext_additional"]) == 1
            ):
                additional_arguments_lookup = (
                    benchmark_handler.generate_additional_argument_lookup(benchmark)
                )

            if need_dynamic_arguments(task):
                dynamic_files_lookup = benchmark_handler.generate_dynamic_file_lookup(
                    benchmark
                )

                _check_dynamic_files_lookup(dynamic_files_lookup)
            print(f"  +Solver:")
            for solver in solver_list:

                solver_options = (
                    config.solver_options.get(solver["id"])
                    if config.solver_options
                    else None
                )

                format = get_accepted_format(solver["format"], benchmark["format"])
                if format is not None:
                    if (
                        need_additional_arguments(task)
                        and len(benchmark_info["benchmark_ext_additional"]) > 1
                    ):
                        additional_arguments_lookup = (
                            benchmark_handler.generate_additional_argument_lookup(
                                benchmark, format
                            )
                        )
                    if task in solver["tasks"]:
                        instances = benchmark_handler.get_instances(
                            benchmark["path"], format
                        )
                        if config.save_output:
                            solver_output_dir = os.path.join(
                                cfg_experiment_result_directory,
                                solver["name"],
                                task,
                                benchmark_info["benchmark_name"],
                            )
                            os.makedirs(solver_output_dir, exist_ok=True)
                        else:
                            solver_output_dir = None

                        for rep in range(1, config.repetitions + 1):
                            desc = f"    {solver['name']}|REP#{rep}"
                            for instance in tqdm(instances, desc=desc):
                                result = solver_handler.run_solver(
                                    solver,
                                    task,
                                    config.timeout,
                                    instance,
                                    format,
                                    additional_arguments_lookup,
                                    dynamic_files_lookup,
                                    output_file_dir=solver_output_dir,
                                    repetition=rep,
                                    solver_options=solver_options,
                                )
                                result.update(benchmark_info)
                                result["repetition"] = rep
                                result["tag"] = config.name
                                write_result(result, result_path, config.result_format)
                                if rep == 1:
                                    Status.increment_instances_counter(
                                        config, task, solver["id"]
                                    )
                else:
                    print(
                        f"    {solver['name']} SKIPPED! No files in supported solver format: {','.join(solver['format'])}"
                    )
        Status.increment_task_counter()
    print("")
    set_experiment_status(
        definitions.EXPERIMENT_INDEX, experiment_id, ExperimentStatus.FINISHED
    )


def _check_dynamic_files_lookup(dynamic_files_lookup):
    missing = []

    for format, instances in dynamic_files_lookup.items():
        if not instances:
            missing.append(format)

    if missing:
        print(
            f"No modification files found for instances in format: {','.join(missing)}"
        )
        exit()


def copy_raws(config: config_handler.Config):
    os.makedirs(config.save_to, exist_ok=True)
    shutil.copy(config.raw_results_path, config.save_to)


def get_last_experiment():
    if os.path.exists(definitions.EXPERIMENT_INDEX):
        experiment_index = pd.read_csv(definitions.EXPERIMENT_INDEX)
        return experiment_index.iloc[-1]

    else:
        return None


def print_experiment_index(tablefmt=None):
    if os.path.exists(definitions.EXPERIMENT_INDEX):
        experiment_index = pd.read_csv(definitions.EXPERIMENT_INDEX)
        print(
            tabulate.tabulate(
                experiment_index, headers="keys", tablefmt=tablefmt, showindex=False
            )
        )

    else:
        print("No experiments found.")


def set_config_of_last_experiment(cfg: config_handler.Config):
    last_experiment = get_last_experiment()
    if last_experiment is None:
        print("No experiment found.")
        exit()
    else:
        user_cfg_yaml = config_handler.load_config_yaml(last_experiment["config_path"])
        cfg.merge_user_input(user_cfg_yaml)
