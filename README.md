# Probo2 - Evaluation tool for abstract argumentation solvers
A python tool that bundles all functionalities needed to collect, validate, analyze, and represent data in the context of benchmarking argumentation solvers.


## Table of Contents
1.[Setup](#setup)

2.[Commands](#commands)

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

4. Install setup.py
 ```
python setup.py install
```
**Note**: If installation fails with this command, try the following command:
```
pip install -e .
```

## Commands

- [Probo2 - Evaluation tool for abstract argumentation solvers](#probo2---evaluation-tool-for-abstract-argumentation-solvers)
  - [Table of Contents](#table-of-contents)
  - [Setup](#setup)
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

### add-solver
Usage: *probo2 add-solver [OPTIONS]*

  Add a solver to the database.

**Options**:
+ *-n, --name*

    Name of the solver  [required]
+ *-p, --path PATH*

    Path to solver executable  [required]
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
  DC/DS problems with the --random_arguments/-rnd option.

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


