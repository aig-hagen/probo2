# Probo2 - Evaluation tool for abstract argumentation solvers

![](src/probo2_data/ressources/probo2_logo.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A python tool that bundles all functionalities needed to collect, validate, analyze, and represent data in the context of benchmarking argumentation solvers.

Click [here](https://aig-hagen.github.io/probo2/) for detailed documentation.
## Changelog

### v1.1 - 22.03.2023
+ **add-solver** command:
    - Interface test checks all supported tasks and formats of solver
    - ICCMA23 instance format (.i23) added to solver interface check
    - Added progressbars to interface check
    - Name is derived from path if not specified

+ **kwt-gen** command:
  - Added instance generation via a config file ( see /generators/generator_configs/kwt_example.yaml)
  - Added parsing of generated instances to ICCMA23 (.i23) format
  - Added option to generate random query arguments for DS an DC tasks
  - Added option to add generated instances as a benchmark to the database
  
+ **quick** command:
    - Added new command to get a quick result for a single instance
+ **board** command:
    - Added new command to create a dashboard for result visualization

+ Added calculation of node and edge homophily  

## Table of Contents
1.[Setup](#setup)

2.[Experiments](#experiments)

3.[Commands](#commands)

## Setup
1. Clone repository
2. Navigate to project folder and create a virtual enviroment
```
python3 -m venv probo2_env
```
3. Activate enviroment
 ```
source probo2_env/bin/activate
```

4. Install
```
pip install -e .
```
## Experiments
In this section we describe the general workflow of probo2 and how experiments are managed.
In probo2, a configuration file fully describes experiments. It contains information about which solvers to run, which benchmark to use, how the results should be validated, what statistics should be calculated, and the graphical representation of the results. This standardized pipeline allows easy and scaleable experiment management and reliably reproducing results.  All configuration options and their default values can be found in the [default_config.yaml](https://github.com/aig-hagen/probo2/blob/working/src/configs_experiment/default_config.yaml) file. The file format of the configuration files is YAML. For further information on YAML files see the offical [specification](https://yaml.org/spec/1.2.2/). Next, we will take a closer look at the [default_config.yaml](https://github.com/aig-hagen/probo2/blob/working/src/configs_experiment/default_config.yaml) file and how probo2 handles configuration files in general.

### Configuration Files
As mentioned before the [default_config.yaml](https://github.com/aig-hagen/probo2/blob/working/src/configs_experiment/default_config.yaml) file contains all default configurations. User specifications (via command line or custom configuration files) overwrite the corresponding default configurations. To avoid conflicts between specifications from the command line and those in a (custom) configuration file, the specifications made by the user via the command line have priority.
If the user does not specify any other options, the experiment will be performed with the default configurations. However, we recommend creating a separate configuration file for each experiment. In general, options are specified as *key:value* pairs in YAML files. The following options can be specified :

+ *name*

   Name/tag of the experiment.

+ *task*

   Comma-separated list of computational tasks to solve.

+ *solver*

   Comma-seperated list of ids or names of solvers to run.

+ *benchmark*

   Comma-separated list of ids or names of benchmarks to run solvers on.


+ *timeout*

   Instance cut-off value in seconds. If cut-off is exceeded instance is marked as timed out. (Default: 600)

+ *repetitions*


   Specifies how often an instance should be repeated. (Default: 1)

+ *result_format*:

   File format for results. (Default: csv)

+ *plot*

   Comma-separated list of kinds of plots to be created.

+ *statistics*

   Comma-separated list of statistics to be calculated.


+ *printing*


   Formatting of the command line output. (Default: 'default')

+ *save_to*

   Directory to store analysis results in. If not specified, the current working directory is used.

+ *copy_raws*

   Copy raws result files to *save_to* destination. (Default: True)

+ *table_export*

   Comma-separated list of export format for tables.

For a list of choices for an option, run the following command:
 
 ```
probo2 run --help
```
 

**Note**: The list is incomplete and constantly being expanded as probo2 is still under development.

### Example

This is an example of a configuration file:

```
name: my_experiment
task: ['DS-PR']
solver: [1,2,'my_solver']
benchmark: all
timeout: 600
repetitions: 3
plot: ['cactus']
statistics: 
- mean
- solved
- timeouts
- coverage
```

Here an experiment named "my_experiment" is configured. The solvers with the ids 1 and 2 and the solver with the name "my_solver" should be executed. In addition, all benchmarks should be used and the 'DS-PR' task should be solved. Each instance shall be repeated 3 times and the cut-off per instance is 600 seconds. After that, the results should be visualized using cactus plots. Since no path was specified by the *save_to* option, the plots are saved in a folder "my_experiment/plots" in the current working directory. In addition, the raw data will also be copied to the folder since the *copy_raws* option is true by default. Furthermore various statistics are calculated. The "statistics" option also shows an alternative syntax for lists.

To run the experiment simply execute the following command:

 ```
probo2 run --config /path/to/my_config.yaml
```

If you want to change the configuration on the fly without changing the file again, we can do it from the command line. For example, if you want to calculate additional statistics ( sum of runtimes) just add the following Options:

 ```
probo2 run --config /path/to/my_config.yaml --statistics sum
```


Another example can be found in the [example_config.yaml](https://github.com/aig-hagen/probo2/blob/working/src/configs_experiment/example_config.yaml)
 

## Commands

- [Probo2 - Evaluation tool for abstract argumentation solvers](#probo2---evaluation-tool-for-abstract-argumentation-solvers)
  - [Changelog](#changelog)
    - [v1.1 - 22.03.2023](#v11---22032023)
  - [Table of Contents](#table-of-contents)
  - [Setup](#setup)
  - [Experiments](#experiments)
    - [Configuration Files](#configuration-files)
    - [Example](#example)
  - [Commands](#commands)
    - [add-solver](#add-solver)
    - [solvers](#solvers)
    - [delete-solver](#delete-solver)
    - [add-benchmark](#add-benchmark)
    - [benchmarks](#benchmarks)
    - [delete-benchmark](#delete-benchmark)
    - [run](#run)
    - [status](#status)
    - [last](#last)
    - [experiment-info](#experiment-info)
    - [plot](#plot)
    - [calculate](#calculate)
    - [validate](#validate)
    - [significance](#significance)
    - [board](#board)

### add-solver
Usage: *probo2 add-solver [OPTIONS]*

  Add a solver to the database.

**Options**:
+ *-n, --name*

    Name of the solver  [required]
+ *-p, --path PATH*

    Path to solver executable  [required].

    Relative paths are automatically resolved. The executable can be a compiled binary, bash script or a python file. As long as the executable implements the ICCMA interface it should work.
    More information on the [ICCMA interface](http://argumentationcompetition.org/).
+ *-f, --format [apx|tgf]*

    Supported format of solver.
+ *-t, --tasks*

    Supported computational problems
+ *-v, --version*

    Version of solver [required].

    This option has to be specified to make sure the same solver with different versions can be added to the database.
+ *-g, --guess*

    Pull supported file format and computational problems from solver.
+ *--help*

    Show this message and exit.

**Example**
```
probo2 add-solver --name MyAwesomeSolver --version 1.23 --path ./path/to/MyAwesomeSolver --guess
```

### solvers
Prints solvers in database to console.

**Options**
  + *-v, --verbose*
  + *--id*

    Print summary of solver with specified id
  + *--help*

    Show this message and exit.


### delete-solver
Usage: *probo2 delete-solver [OPTIONS]*

  Deletes a solver from the database. Deleting has to be confirmed by user.
**Options**:
+ *--id*

    ID of solver to delete.
+ *--all*

    Delete all solvers in database.
+ *--help*

    Show this message and exit.


### add-benchmark
Usage: *probo2 add-benchmark [OPTIONS]*

  Adds a benchmark to the database. Before a benchmark is added to the
  database, it is checked if each instance is present in all specified file
  formats. Missing instances can be generated after the completion test
  (user has to confirm generation) or beforehand via the --generate/-g
  option. It is also possilbe to generate random argument files for the
  DC/DS problems with the --random_arguments/-rnd option. By default, the following attributes are saved for a benchmark: name, path, format, and the extension of query argument files. However, it is possible to specify additional attributes using your functions. See section "custom function" for further information.
  If no benchmark name via the --name option is provided, the name is derived from the benchmark path. Instance formats and the file extension of the query arguments (used for DS and DC tasks) are automatically set if not specified. For this the file extensions of all files in the given path are compared with the default file formats\extensions (see src/utils/definitions.DefaultInstanceFormats and src/utils/definitions.DefaultQueryFormats). Formats are set to the intesection between found formats and default formats. 

**Options**:
+ *-n, --name *

    Name of benchmark/fileset  [required]
+ *-p, --path*

    Directory of instances [required]
    Subdirectories are recursively searched for instances.


+ *-f, --format [apx|tgf]*

    Supported formats of benchmark/fileset  [required]

+ *-ext, --extension_arg_files*

    Extension of additional argument parameter for DC/DS problems.
    Default is "arg"
+ *--no_check*

    Checks if the benchmark is complete.
+ *-g, --generate [apx|tgf]*

    Generate instances in specified format
+ *-rnd, --random_arguments*

    Generate additional argument files with a random argument.
+ *-fun, --function*

    Custom functions to add additional attributes to benchmark.
+ *--help*

    Show this message and exit.
 

**Example**
```
probo2 add-benchmark --name MyTrickyBenchmark --path ./path/to/MyTrickyBenchmark --format tgf --generate apx -rnd
```


### benchmarks

  Prints benchmarks in database to console.


**Options**

  + *-v, --verbose*

    Prints additional information on benchmark
  + *--help*

    Show this message and exit.


### delete-benchmark
Usage: *probo2 delete-benchmark [OPTIONS]*

  Deletes a benchmark from the database. Deleting has to be confirmed by
  user.

**Options**:
+ *--id*

    ID of benchmark to delete.
+ *--all*

    Delete all benchmarks in database
+ *--help*

    Show this message and exit.


### run

  Run solver.

**Options**
  + *-a, --all*

    Execute all solvers supporting the specified tasks on specified instances.

  + *-slct, --select*

    Execute (via solver option) selected solver supporting the specified tasks.

  + *-s, --solver*

    Comma-seperated list of ids or names of solvers (in database) to run.

  + *-b, --benchmark*

    Comma-seperated list of ids or names of benchmarks (in database) to run solvers on.

  + *--task*

    Comma-seperated list of tasks to solve.
  + *-t, --timeout*

    Instance cut-off value in seconds. If cut-off is exceeded instance is marked as timed out.

  + *--dry*

    Print results to command-line without saving to the database.

  + *--track*

    Comma-seperated list of tracks to solve.
  + *--tag*

    Tag for individual experiments.This tag is used to identify the experiment.  [required]

  + *--notify*

    Send a notification to the email address provided as soon as the experiments are finished.

  + *-n, --n_times*

    Number of repetitions per instance. Run time is the avg of the n runs.

  + *-sub, --subset*

    Run only the first n instances of a benchmark.

  + *--multi*

    Run experiment on mutiple CPU cores. The number of cores to use is #physical cores - 1 or 1. This is a heuristic to avoid locking up the system.


  + *--help*

    Show this message and exit.
**Example**
```
probo2 run --all --benchmark my_benchmark --task EE-CO,EE-PR --tag MyExperiment --timeout 600 --notify my@mail.de
```


### status
Usage: *probo2 status*

  Provides an overview of the progress of the currently running experiment.

**Options**:
+ *--help*

    Show this message and exit.


### last
Usage: *probo2 last*

  Shows basic information about the last finished experiment.

  Infos include experiment tag, benchmark names, solved tasks, executed
  solvers and and the time when the experiment was finished

**Options**:
+ *--help*

    Show this message and exit.


### experiment-info
Usage: *probo2 experiment-info [OPTIONS]*

  Prints some basic information about the experiment speficied with "--tag"
  option.

**Options**:
+ *-t, --tag*

    Experiment tag.  [required]
+ *--help*

    Show this message and exit.


### plot
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
+ *-t, --tag*

     Comma-separated list of experiment tags to be selected.
+ *--task*

    Comma-separated list of task IDs or symbols to be selected.
+ *--benchmark*

    Comma-separated list of benchmark IDs or names to be selected.
+ *-s, --solver*

    Comma-separated list of solver IDs or names to be selected.
+ *-st, --save_to*

    Directory to store plots in. Filenames will be generated automatically.
+ *--vbs*

    Create virtual best solver from experiment data.
+ *-b, --backend*

    Backend to use.
    Choices: [pdf|pgf|png|ps|svg]
+ *-c, --combine*

    Combine results on specified key
    Choices: [tag|task_id|benchmark_id]
+ *-k, --kind*

    Kind of plot to create:
    Choices: [cactus|count|dist|pie|box|all]

+ *--compress*

    Compress saved files.
    Choices: [tar|zip]
+ *-s, --send*

    Send plots via E-Mail
+ *-l, --last*

    Plot results for the last finished experiment.
+ *--help*

    Show this message and exit.

**Example**
```
probo2 plot --kind cactus --tag MyExperiment --compress zip --send my@mail.de
```


### calculate
Calculte different statistical measurements.
Usage: probo2 calculate [OPTIONS]

**Options**
 + *-t, --tag*

     Comma-separated list of experiment tags to be selected.
+ *--task*

    Comma-separated list of task IDs or symbols to be selected.
+ *--benchmark*

    Comma-separated list of benchmark IDs or names to be selected.
+ *-s, --solver*

    Comma-separated list of solver IDs or names to be selected.

+ *-p, --par*

    Penalty multiplier for PAR score
+ *-s, --solver*
    Comma-separated list of solver ids or names.

+ *-pfmt, --print_format*

    Table format for printing to console.

    Choices: [plain|simple|github|grid|fancy_grid|pipe|orgtbl|jira|presto|pretty|psql|rst|mediawiki|moinmoin|youtrack|html|unsafehtmllatex|latex_raw|latex_booktabs|textile]

+ *-c, --combine*

    Combine results on key.

    Choices: [task_id|benchmark_id|solver_id]
+ *--vbs*

    Create virtual best solver

+ *-st, --save_to*

    Directory to store tables.

+ *-e, --export*

    Export results in specified format.

    Choices: [latex|json|csv]
+ *-s, --statistics*

    Stats to calculate.

    Choices:[mean|sum|min|max|median|var|std|coverage|timeouts|solved|errors|all]
+ *-l, --last*

    Calculate stats for the last finished experiment.

+ *--compress*

    Compress saved files.
    Choices: [tar|zip]
+ *-s, --send*

    Send files via E-Mail

+ *--help*

  Show this message and exit.

**Example**
```
probo2 calculate --tag MyExperiment -s timeouts -s errors -s solved --par 10 --compress zip --send my@mail.de
```


### validate
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
+ *--tag*

    Experiment tag to be validated
+ *-t, --task*

    Comma-separated list of task IDs or symbols to be validated.
+ *-b, --benchmark*

    Benchmark name or id to be validated.  [required]
+ *-s, --solver*

    Comma-separated list of solver IDs or names to be validated.  [required]
+ *-f, --filter*

    Filter results in database. Format: [column:value]
+ *-r, --reference PATH*

    Path to reference files.
+ *--update_db*

    Update instances status (correct, incorrect, no reference) in database.
+ *-pw, --pairwise*

    Pairwise comparision of results. Not supported for the SE task.
+ *-e, --export*

    Export results in specified format.
    Choices:  [json|latex|csv]
+ *--raw*

    Export raw validation results in csv format.
+ *-p, --plot*

    Create a heatmap for pairwise comparision results and a count plot for validation with references.
    Choices:  [heatmap|count]
+ *-st, --save_to*

    Directory to store plots and data in. Filenames will be generated automatically.
+ *-ext, --extension*

    Reference file extension
+ *--compress*

    Compress saved files.
    Choices:  [tar|zip]
+ *--send*

    Send plots and data via E-Mail.
+ *-v, --verbose*

    Verbose output for validation with reference. For each solver the instance names of not validated and incorrect instances is printed to the console.
+ *--help*

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

### significance
Usage: *probo2 significance [OPTIONS]*

  Parmatric and non-parametric significance and post-hoc tests.

**Options**:
+ *--tag*

    Experiment tag to be tested.
+ *-t, --task*

    Comma-separated list of task IDs or symbols to be tested.
+ *--benchmark*

    Benchmark name or id to be tested.
+ *-s, --solver*

    Comma-separated list of solver id.s  [required]

+ *-p, --parametric*

    Parametric significance test. ANOVA for mutiple solvers and t-test for two solvers.
    Choices: [ANOVA|t-test]
+ *-np, --non_parametric*

    Non-parametric significance test. kruksal for mutiple solvers and mann-whitney-u for two solvers
    Choices:  [kruskal|mann-whitney-u]

+ *-php, --post_hoc_parametric*

    Parametric post-hoc tests.
    Choices:  [scheffe|tamhane|ttest|tukey|tukey_hsd]

+ *-phn, --post_hoc_non_parametric*

    Non-parametric post-hoc tests.
    Choices:  [conover|dscf|mannwhitney|nemenyi|dunn|npm_test|vanwaerden|wilcoxon]

+ *-a, --alpha FLOAT*

    Significance level.

+ *-l, --last*

    Test the last finished experiment.
+ *--help*

    Show this message and exit.

**Example**

```
probo2 significance --tag MyExperiment --parametric ANOVA --php scheffe
```
### board
Probo2 provides an interactive dashboard to visualize results of experiments. The dashboard contains plots and tables which can be filtered using checkboxes in the sidebar.
![](src/probo2_data/ressources/probo2Board.gif)

Usage: *probo2 board [OPTIONS]*

  Launch dashboard for experiment visualization.

**Options**:

+ *--tag, -t*

    Experiment tag
+ *--raw, -r*
  
  Full path to a raw results file  (raw.csv).

  **Note**: Only needed when no tag is specified.
