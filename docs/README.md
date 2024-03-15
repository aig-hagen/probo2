# Probo2

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Probo2 is an end-to-end benchmark framework for abstract argumentation solvers. This framework aims at providing researchers
and developers with an easy-to-use, robust, and flexible command-line tool to simplify
and speed up typical tasks in benchmarking abstract argumentation solvers. Therefore,
probo2 offers functionalities to (1) generate benchmarks, (2) execute solvers and collect data, (3) verify the correctness of solvers, (4) compare the performance of solvers, (5)
perform statistical analysis and—of particular interest in the research context—(6) generate "publication-ready" visualizations of results. probo2 offers a standardized pipeline
for evaluating argumentation solvers. Users can specify which performance metrics to
use and how to visualize the results, allowing to compare different solvers easily. To reproduce results reliably, any experiment is completely described by a configuration file.
In addition, probo2 supports the parallelization of experiments. During development,
we set a strong focus on customizability. The modular design of probo2 allows users
to modify and extend functionality to their needs.

The following sections will guide you through the initial steps of installing probo2 and running your first benchmarking experiment. By the end of this guide, you'll have a solid understanding of the probo2 workflow and can start experimenting.

## Installation

- Clone the repository

```
git clone https://github.com/aig-hagen/probo2.git
```

- Navigate to project folder and create a virtual enviroment

```bash
python3 -m venv probo2_env
```

or if use Anaconda:

```bash
conda create -n probo2_env pip
```

- Activate the created enviroment

 ```bash
source probo2_env/bin/activate
```

- Install probo2

```bash
pip install
```

By installing probo2 in editable mode in your environment, you can ensure that any changes you make to probo2 are immediately available.

```bash
pip install -e .
```

## Run your first experiment

In this section, we describe the general workflow of probo2 and how experiments are managed.
As you will see, with probo2, it is easy to set up an experiment. We only need three things:
>
> 1. The argumentation solvers
> 2. The benchmark instances
> 3. A configuration file

You can manually add solvers and benchmarks with the [add-solver](#add-solver) or [add-benchmark](#add-benchmark) command.
For our first experiment, however, we will use a very convenient functionality, the [fetch](#fetch) command.
With this command, we can automatically download and install the [ICCMA](http://argumentationcompetition.org/). solvers and benchmarks and add them to probo2. We have to specify which solvers we want to install, which benchmarks we want to load, and where they should be saved. For example, if we want to have the __ICCMA23__ solvers and benchmarks, just run:

```bash
probo2 fetch --benchmark ICCMA23 --solver ICCMA23 --install --save_to /path/to/save
```

You can use the [solvers](#solvers) command to verify that all solvers have been properly installed and added to the probo2 database. Just run:

```bash
probo2 solvers 
```
The output on the console should look something like this:
```
+----+-------------------+---------+-----------------------+
| id |       name        | version |        format         |
+----+-------------------+---------+-----------------------+
| 1  | mu-toksia-crpyto  | ICCMA23 | ['apx', 'tgf','i23]   |
| 2  |      Fudge        | ICCMA23 | ['apx', 'tgf','i23]   |
| 3  | mu-toksia-glucose | ICCMA23 | ['apx', 'tgf', 'i23'] |
| 4  |     crustabri     | ICCMA23 |        ['i23']        |
| 5  |      PORTSAT      | ICCMA23 |        ['i23']        |
+----+-------------------+---------+-----------------------+
```

You can also display the existing [benchmarks](#benchmarks) with

```bash
probo2 benchmarks
```

Now we just need one more thing: A Configuration file.
In probo2, a configuration file fully describes experiments. It contains information about which solvers to run, which benchmark to use, how the results should be validated, what statistics should be calculated, and the graphical representation of the results. This standardized pipeline allows easy and scaleable experiment management and reliably reproducing results.  All configuration options and their default values can be found in the [default_config.yaml](https://github.com/aig-hagen/probo2/blob/working/src/configs_experiment/default_config.yaml) file. The file format of the configuration files is YAML. For further information on YAML files see the offical [specification](https://yaml.org/spec/1.2.2/).

Since Probo2 comes with many useful preset options, we need only specify a few for our first experiment. This is what the config file looks like:

```
name: my_first_experiment
task: ['DS-PR']
solver: all
benchmark: all
timeout: 1
```


## Configuration Files

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

## Example

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
