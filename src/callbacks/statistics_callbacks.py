
import pandas as pd
from typing import Any
from hydra.experimental.callback import Callback
from omegaconf import DictConfig



class All(Callback):
    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        result_file = config['combined_results_file']
        df = pd.read_csv(result_file)
        # Filter out rows where 'exit_with_error' is True
        filtered_data = df[df['exit_with_error'] == False]
        
        # Group by solvername and calculate statistics
        stats = filtered_data.groupby(['task','benchmark_name','solver_name']).agg(
            total_runs=('run', 'count'),
            mean_perfcounter_time=('perfcounter_time', 'mean'),
            mean_user_sys_time=('user_sys_time', 'mean'),
            num_timed_out=('timed_out', lambda x: x.sum()),
            num_not_timed_out=('timed_out', lambda x: (~x).sum()),
            proportion_timed_out=('timed_out', 'mean')
        ).reset_index()

        print(stats)

class MeanRuntime(Callback):
    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        result_file = config['combined_results_file']
        df = pd.read_csv(result_file)
        # Filter out rows where 'exit_with_error' is True
        filtered_data = df[df['exit_with_error'] == False]
        
        # Group by solvername and calculate statistics
        stats = filtered_data.groupby(['task','benchmark_name','solver_name']).agg(
            mean_perfcounter_time=('perfcounter_time', 'mean'),
            mean_user_sys_time=('user_sys_time', 'mean'),
        ).reset_index()

        print(stats)

class Coverage(Callback):
    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        result_file = config['combined_results_file']
        df = pd.read_csv(result_file)
        # Filter out rows where 'exit_with_error' is True
        filtered_data = df[df['exit_with_error'] == False]
        
        # Group by solvername and calculate statistics
        stats = filtered_data.groupby(['task','benchmark_name','solver_name']).agg(
            coverage=('timed_out',  lambda x: (~x).mean())
        ).reset_index()

        print(stats)
        stats.to_csv(config.csv_stats_result_file)





