# Implementation Details

## Running Solver with Arguments
To run solver with Arguments one must speficy the arguments in the solver yaml.
At runtime it is check if the arguments are present and appended to the solver call.

## Saving and Aggregation of Stats and Scores
A Evaluation folder is created.
Each stat is written to its own csv file. The paths to this files are written to a index file. After all stats are calculated the dataframes are loaded and aggregated grouped by the benchmark and task. The combined dataframe is saved

## Table Exporting
The results of the evaluation can be exported as tables of different formats.



# TODO

- Wenn solver mit paramtern aufgerufen werden müssen die parameter an den solver namen angehängt werden - done
-  Table export pipline - done
   -  Text - done
   -  Latex
- Add suport for DC and DC tasks - Done

- Before call backs are run, check if the path all exits - DONE
- Check if for all instances query arguments are present in benchmark validation - DONE
- Implement solver config validation: - Done
  - Check if path exists - Done

- Check if output of solver is valid dependeing on the interface and Task -> if not treat as exited_with error and set time to 600


- Add a include errors options to the coverage callback

- ** Implementation of result validation **
- Add a correct_solution flag, which is true until the validation changes it



## Planned Feature

- Pandasai integration to describe tables
- Notification integration
