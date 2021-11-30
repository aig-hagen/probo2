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
## Commands
### add-solver
Adds a solver to the database.

**Options**
+ *--name/-n* (required):

    Name of the solver you want to add to the database.

+ *--path/-p* (required):

    Path to the solvers executable. Relative paths are automatically resolved.
    The executable can be a compiled binary, bash script or a python file.

+  *--format/-f*:

    Supported file format of the solver.

+ *--tasks/-t*:

    Comma-seperated list of supported computational problems.

+ *--version/-v* (required):

    Solver version. This option has to be specified to make sure the same solver with different versions can be added to the database.

+ *--guess/-g*:

    Pull supported file format and computational problems from solver.


