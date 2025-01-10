# Implementation Details

## Running Solver with Arguments
To run solver with Arguments one must speficy the arguments in the solver yaml.
At runtime it is check if the arguments are present and appended to the solver call.

## Saving and Aggregation of Stats and Scores
A Evaluation folder is created.
Each stat is written to its own csv file. The paths to this files are written to a index file. After all stats are calculated the dataframes are loaded and aggregated grouped by the benchmark and task. The combined dataframe is saved

## Table printing