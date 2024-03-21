## add-Solver

The `add-solver` command is used for adding solvers to the database.
.Below is a detailed documentation and tutorial on how to use the `add-solver` command effectively.

### Command Syntax

```shell
probo2 add_solver [OPTIONS]
```

### Options

### `--name`, `-n`

- **Description**: Specifies the name of the solver.
- **Required**: No
- **Example**: `-n "MySolver"`

### `--path`, `-p`

- **Description**: Path to the solver executable.
Relative paths are automatically resolved. The executable can be a compiled binary, bash script or a python file. As long as the executable implements the ICCMA interface it should work.
More information on the [ICCMA interface](http://argumentationcompetition.org/).
- **Required**: Yes
- **Type**: `click.Path`
- **Constraints**: The path must exist and be resolvable.
- **Examples**:
  - `-p /usr/local/bin/mysolver`
  - `--path ./mysolver`
  
### `--format`, `-f`

- **Description**: Defines the supported file formats of the solver. This option can be specified multiple times if the solver supports more than one format.
- **Required**: No
- **Multiple**: Yes
- **Example**: `-f "tgf" -f "apx"`

### `--tasks`, `-t`

- **Description**: Lists the supported computational problems/tasks. This is checked against predefined tasks to ensure compatibility.
- **Required**: No
- **Default**: `[]`
- **Callback**: `CustomClickOptions.check_problems`
- **Example**: `-t "Optimization" -t "Satisfaction"`

### `--version`, `-v`

- **Description**: Indicates the version of the solver.
- **Required**: No
- **Type**: `String`
- **Example**: `-v "1.0.3"`

### `--fetch`, `-ft`

- **Description**: When set, Probo2 will attempt to pull the supported file format and computational problems directly from the solver. This is a flag option.
- **Required**: No
- **Type**: `Flag`
- **Example**: `--fetch`

### `--yes`

- **Description**: Skips the confirmation prompt when adding a solver to the database. This is a convenience flag for automation or scripting.
- **Required**: No
- **Type**: `Flag`
- **Example**: `--yes`

### `--no_check`

- **Description**: Disables the validation check of solver's properties. Use with caution.
- **Required**: No
- **Type**: `Flag`
- **Example**: `--no_check`

### Example Calls

#### Adding a Solver with Minimum Required Information

```shell
probo2 add_solver -p "/usr/local/bin/mysolver"
```

#### Adding a Solver with Full Information

```shell
probo2 add_solver -n "MySolver" -p "/usr/local/bin/mysolver" -f "XML" -f "JSON" -t "Optimization" -v "1.0.3" --yes
```

#### Fetching Supported Formats and Tasks Automatically

```shell
probo2 add_solver -n "AutoSolver" -p "/opt/solvers/autosolver" --fetch --yes
```

### Notes

- When specifying multiple formats or tasks, repeat the option for each value.
- Use the `--fetch` option to automatically retrieve solver capabilities, if the solver supports this feature.
- The `--yes` option is particularly useful for scripting or when running in environments where user interaction is not possible.
- Ensure the path to the solver executable is correct to avoid errors during the addition process.

## add-solver

Usage: *probo2 add-solver [OPTIONS]*

  Add a solver to the database.

**Options**:

- *-n, --name*

    Name of the solver  [required]
- *-p, --path PATH*

    Path to solver executable  [required].

    Relative paths are automatically resolved. The executable can be a compiled binary, bash script or a python file. As long as the executable implements the ICCMA interface it should work.
    More information on the [ICCMA interface](http://argumentationcompetition.org/).
- *-f, --format [apx|tgf]*

    Supported format of solver.
- *-t, --tasks*

    Supported computational problems
- *-v, --version*

    Version of solver [required].

    This option has to be specified to make sure the same solver with different versions can be added to the database.
- *-g, --guess*

    Pull supported file format and computational problems from solver.
- *--help*

    Show this message and exit.

**Example**

```
probo2 add-solver --name MyAwesomeSolver --version 1.23 --path ./path/to/MyAwesomeSolver --guess
```

## fetch

## solvers

Prints solvers in database to console.

**Options**

- *-v, --verbose*
- *--id*

    Print summary of solver with specified id
- *--help*

    Show this message and exit.

## delete-solver

Usage: *probo2 delete-solver [OPTIONS]*

  Deletes a solver from the database. Deleting has to be confirmed by user.
**Options**:

- *--id*

    ID of solver to delete.
- *--all*

    Delete all solvers in database.
- *--help*

    Show this message and exit.

## add-benchmark

Usage: *probo2 add-benchmark [OPTIONS]*

  Adds a benchmark to the database. Before a benchmark is added to the
  database, it is checked if each instance is present in all specified file
  formats. Missing instances can be generated after the completion test
  (user has to confirm generation) or beforehand via the --generate/-g
  option. It is also possilbe to generate random argument files for the
  DC/DS problems with the --random_arguments/-rnd option. By default, the following attributes are saved for a benchmark: name, path, format, and the extension of query argument files. However, it is possible to specify additional attributes using your functions. See section "custom function" for further information.
  If no benchmark name via the --name option is provided, the name is derived from the benchmark path. Instance formats and the file extension of the query arguments (used for DS and DC tasks) are automatically set if not specified. For this the file extensions of all files in the given path are compared with the default file formats\extensions (see src/utils/definitions.DefaultInstanceFormats and src/utils/definitions.DefaultQueryFormats). Formats are set to the intesection between found formats and default formats.

**Options**:

- *-n, --name*

    Name of benchmark/fileset  [required]
- *-p, --path*

    Directory of instances [required]
    Subdirectories are recursively searched for instances.

- *-f, --format [apx|tgf]*

    Supported formats of benchmark/fileset  [required]

- *-ext, --extension_arg_files*

    Extension of additional argument parameter for DC/DS problems.
    Default is "arg"
- *--no_check*

    Checks if the benchmark is complete.
- *-g, --generate [apx|tgf]*

    Generate instances in specified format
- *-rnd, --random_arguments*

    Generate additional argument files with a random argument.
- *-fun, --function*

    Custom functions to add additional attributes to benchmark.
- *--help*

    Show this message and exit.

**Example**

```
probo2 add-benchmark --name MyTrickyBenchmark --path ./path/to/MyTrickyBenchmark --format tgf --generate apx -rnd
```

## benchmarks

  Prints benchmarks in database to console.

**Options**

- *-v, --verbose*

    Prints additional information on benchmark
- *--help*

    Show this message and exit.

## delete-benchmark

Usage: *probo2 delete-benchmark [OPTIONS]*

  Deletes a benchmark from the database. Deleting has to be confirmed by
  user.

**Options**:

- *--id*

    ID of benchmark to delete.
- *--all*

    Delete all benchmarks in database
- *--help*

    Show this message and exit.


## convert-benchmark

The `convert-benchmark` is used for the conversion of benchmarks into different formats, potentially including query argument files. This documentation details how to use the `convert-benchmark` command.

### Command Syntax

```shell
probo2 convert-benchmark [OPTIONS]
```

### Options

### `--id`
- **Description**: The unique identifier of the benchmark to be converted.
- **Type**: Integer
- **Required**: Yes
- **Example**: `--id 3`

### `--name`, `-n`
- **Description**: The name for the newly generated benchmark. If not specified, a format suffix is added to the original name.
- **Type**: String
- **Required**: No
- **Example**: `-n "NewBenchmarkName"`

### `--formats`, `-f`
- **Description**: Specifies the formats to which the selected benchmark will be converted. A separate benchmark is created for each specified format.
- **Type**: String
- **Required**: Yes
- **Multiple**: Yes
- **Example**: `-f i23 -f apx`

### `--save_to`, `-st`
- **Description**: The directory where the converted benchmark will be stored. By default, this is the current working directory.
- **Type**: Path
- **Required**: No
- **Example**: `-st "/path/to/directory"`

### `--add`, `-a`
- **Description**: If set, the generated benchmark will be added to the database.
- **Type**: Boolean
- **Required**: No
- **Is Flag**: True
- **Example**: `-a`

### `--skip_args`, `-s`
- **Description**: Skip the creation of argument files for the converted benchmarks.
- **Type**: Boolean
- **Required**: No
- **Is Flag**: True
- **Example**: `-s`

### Example Calls

### Convert a Benchmark with Minimal Requirements

```shell
probo2 convert-benchmark --id 12 -f i23
```

### Convert and Name the Benchmark, Skipping Argument File Creation

```shell
probo2 convert-benchmark --id 12 -n "CustomBenchmark" -f i23 -s
```

### Convert a Benchmark and Save to a Specified Directory

```shell
probo2 convert-benchmark --id 23 -f apx -f i23 -st "/path/to/save"
```

### Convert a Benchmark, Add to Database, and Skip Argument Files

```shell
probo2 convert-benchmark --id 10 -f i23 -a -s
```

### Additional Notes

- The `--formats` option must be specified at least once but can be repeated to specify multiple formats.
- If the `--save_to` option is not specified, the current working directory is used by default.
- The `--add` flag, if used, indicates that the converted benchmarks should be automatically added to the database.
- The `--skip_args` flag allows users to skip the generation of argument files for benchmarks, useful when these files are not needed or already exist.





## run

  Run solver.

**Options**

- *-a, --all*

    Execute all solvers supporting the specified tasks on specified instances.

- *-slct, --select*

    Execute (via solver option) selected solver supporting the specified tasks.

- *-s, --solver*

    Comma-seperated list of ids or names of solvers (in database) to run.

- *-b, --benchmark*

    Comma-seperated list of ids or names of benchmarks (in database) to run solvers on.

- *--task*

    Comma-seperated list of tasks to solve.
- *-t, --timeout*

    Instance cut-off value in seconds. If cut-off is exceeded instance is marked as timed out.

- *--dry*

    Print results to command-line without saving to the database.

- *--track*

    Comma-seperated list of tracks to solve.
- *--tag*

    Tag for individual experiments.This tag is used to identify the experiment.  [required]

- *--notify*

    Send a notification to the email address provided as soon as the experiments are finished.

- *-n, --n_times*

    Number of repetitions per instance. Run time is the avg of the n runs.

- *-sub, --subset*

    Run only the first n instances of a benchmark.

- *--multi*

    Run experiment on mutiple CPU cores. The number of cores to use is #physical cores - 1 or 1. This is a heuristic to avoid locking up the system.

- *--help*

    Show this message and exit.
**Example**

```
probo2 run --all --benchmark my_benchmark --task EE-CO,EE-PR --tag MyExperiment --timeout 600 --notify my@mail.de
```

## status

Usage: *probo2 status*

  Provides an overview of the progress of the currently running experiment.

**Options**:

- *--help*

    Show this message and exit.

## last

Usage: *probo2 last*

  Shows basic information about the last finished experiment.

  Infos include experiment tag, benchmark names, solved tasks, executed
  solvers and and the time when the experiment was finished

**Options**:

- *--help*

    Show this message and exit.

## experiment-info

Usage: *probo2 experiment-info [OPTIONS]*

  Prints some basic information about the experiment speficied with "--tag"
  option.

**Options**:

- *-t, --tag*

    Experiment tag.  [required]
- *--help*

    Show this message and exit.

## plot

Usage: *probo2 plot[OPTIONS]*

  Create plots of experiment results.

  The --tag option is used to specify which experiment the plots should be
  created for. With the options --solver, --task and --benchmark you can
  further restrict this selection. If only a tag is given, a plot is
  automatically created for each task and benchmark of this experiment. With
  the option --kind you determine what kind of plot should be created. It is
  also possible to combine the results of different experiments, benchmarks
  and tasks with the --combine option.

**Options**:

- *-t, --tag*

     Comma-separated list of experiment tags to be selected.
- *--task*

    Comma-separated list of task IDs or symbols to be selected.
- *--benchmark*

    Comma-separated list of benchmark IDs or names to be selected.
- *-s, --solver*

    Comma-separated list of solver IDs or names to be selected.
- *-st, --save_to*

    Directory to store plots in. Filenames will be generated automatically.
- *--vbs*

    Create virtual best solver from experiment data.
- *-b, --backend*

    Backend to use.
    Choices: [pdf|pgf|png|ps|svg]
- *-c, --combine*

    Combine results on specified key
    Choices: [tag|task_id|benchmark_id]
- *-k, --kind*

    Kind of plot to create:
    Choices: [cactus|count|dist|pie|box|all]

- *--compress*

    Compress saved files.
    Choices: [tar|zip]
- *-s, --send*

    Send plots via E-Mail
- *-l, --last*

    Plot results for the last finished experiment.
- *--help*

    Show this message and exit.

**Example**

```
probo2 plot --kind cactus --tag MyExperiment --compress zip --send my@mail.de
```

## calculate

Calculte different statistical measurements.
Usage: probo2 calculate [OPTIONS]

**Options**

- *-t, --tag*

     Comma-separated list of experiment tags to be selected.
- *--task*

    Comma-separated list of task IDs or symbols to be selected.
- *--benchmark*

    Comma-separated list of benchmark IDs or names to be selected.
- *-s, --solver*

    Comma-separated list of solver IDs or names to be selected.

- *-p, --par*

    Penalty multiplier for PAR score
- *-s, --solver*
    Comma-separated list of solver ids or names.

- *-pfmt, --print_format*

    Table format for printing to console.

    Choices: [plain|simple|github|grid|fancy_grid|pipe|orgtbl|jira|presto|pretty|psql|rst|mediawiki|moinmoin|youtrack|html|unsafehtmllatex|latex_raw|latex_booktabs|textile]

- *-c, --combine*

    Combine results on key.

    Choices: [task_id|benchmark_id|solver_id]
- *--vbs*

    Create virtual best solver

- *-st, --save_to*

    Directory to store tables.

- *-e, --export*

    Export results in specified format.

    Choices: [latex|json|csv]
- *-s, --statistics*

    Stats to calculate.

    Choices:[mean|sum|min|max|median|var|std|coverage|timeouts|solved|errors|all]
- *-l, --last*

    Calculate stats for the last finished experiment.

- *--compress*

    Compress saved files.
    Choices: [tar|zip]
- *-s, --send*

    Send files via E-Mail

- *--help*

  Show this message and exit.

**Example**

```
probo2 calculate --tag MyExperiment -s timeouts -s errors -s solved --par 10 --compress zip --send my@mail.de
```

## validate

Usage: *probo2 validate [OPTIONS]*

  Validate experiments results.

  With the validation, we have the choice between a pairwise comparison of
  the results or a validation based on reference results. Pairwise
  validation is useful when no reference results are available. For each
  solver pair, the instances that were solved by both solvers are first
  identified. The results are then compared and the percentage of accordance
  is calculated and output in the form of a table. It is also possible to
  show the accordance of the different solvers as a heatmap with the "--plot
  heatmap" option. Note: The SE task is not supported here.

  For the validation with references, we need to specify the path to our
  references results with the option "--ref". It's important to note that
  each reference instance has to follow a naming pattern to get matched with
  the correct result instance. The naming of the reference instance has to
  include (1) the full name of the instance to validate, (2) the task, and
  (3) the specified extension. For example for the instance
  "my_instance.apx",  the corresponding reference instance for the EE-PR
  task would be named as follows: "my_instance_EE-PR.out" The order of name
  and task does not matter. The extension is provided via the "--extension"
  option.

**Options**:

- *--tag*

    Experiment tag to be validated
- *-t, --task*

    Comma-separated list of task IDs or symbols to be validated.
- *-b, --benchmark*

    Benchmark name or id to be validated.  [required]
- *-s, --solver*

    Comma-separated list of solver IDs or names to be validated.  [required]
- *-f, --filter*

    Filter results in database. Format: [column:value]
- *-r, --reference PATH*

    Path to reference files.
- *--update_db*

    Update instances status (correct, incorrect, no reference) in database.
- *-pw, --pairwise*

    Pairwise comparision of results. Not supported for the SE task.
- *-e, --export*

    Export results in specified format.
    Choices:  [json|latex|csv]
- *--raw*

    Export raw validation results in csv format.
- *-p, --plot*

    Create a heatmap for pairwise comparision results and a count plot for validation with references.
    Choices:  [heatmap|count]
- *-st, --save_to*

    Directory to store plots and data in. Filenames will be generated automatically.
- *-ext, --extension*

    Reference file extension
- *--compress*

    Compress saved files.
    Choices:  [tar|zip]
- *--send*

    Send plots and data via E-Mail.
- *-v, --verbose*

    Verbose output for validation with reference. For each solver the instance names of not validated and incorrect instances is printed to the console.
- *--help*

    Show this message and exit.

**Example**

Validation with references:

```
probo2 validate --tag MyExperiment --benchmark MyBenchmark --references /path/to/references --export latex --plot count --compress zip --send my@mail.de --save_to .
```

Pairwise validation:

```
probo2 validate --tag MyExperiment --benchmark MyBenchmark --pairwise --plot heatmap --save_to .
```

## significance

Usage: *probo2 significance [OPTIONS]*

  Parmatric and non-parametric significance and post-hoc tests.

**Options**:

- *--tag*

    Experiment tag to be tested.
- *-t, --task*

    Comma-separated list of task IDs or symbols to be tested.
- *--benchmark*

    Benchmark name or id to be tested.
- *-s, --solver*

    Comma-separated list of solver id.s  [required]

- *-p, --parametric*

    Parametric significance test. ANOVA for mutiple solvers and t-test for two solvers.
    Choices: [ANOVA|t-test]
- *-np, --non_parametric*

    Non-parametric significance test. kruksal for mutiple solvers and mann-whitney-u for two solvers
    Choices:  [kruskal|mann-whitney-u]

- *-php, --post_hoc_parametric*

    Parametric post-hoc tests.
    Choices:  [scheffe|tamhane|ttest|tukey|tukey_hsd]

- *-phn, --post_hoc_non_parametric*

    Non-parametric post-hoc tests.
    Choices:  [conover|dscf|mannwhitney|nemenyi|dunn|npm_test|vanwaerden|wilcoxon]

- *-a, --alpha FLOAT*

    Significance level.

- *-l, --last*

    Test the last finished experiment.
- *--help*

    Show this message and exit.

**Example**

```
probo2 significance --tag MyExperiment --parametric ANOVA --php scheffe
```

## board

Probo2 provides an interactive dashboard to visualize results of experiments. The dashboard contains plots and tables which can be filtered using checkboxes in the sidebar.
![](src/probo2_data/ressources/probo2Board.gif)

Usage: *probo2 board [OPTIONS]*

  Launch dashboard for experiment visualization.

**Options**:

- *--tag, -t*

    Experiment tag
- *--raw, -r*
  
  Full path to a raw results file  (raw.csv).

  **Note**: Only needed when no tag is specified.
