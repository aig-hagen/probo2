import src.functions.register as register
import pandas as pd
import tabulate
from src.utils.benchmark_handler import load_benchmark_by_identifier


def _print_results(df: pd.DataFrame, test):
    p = df.p_value.iloc[0]
    benchmark_name = load_benchmark_by_identifier(
        [int(df.benchmark_id.iloc[0])])[0]['name']
    print(f"Task: {df.task.iloc[0]}")
    print(f"Benchmark: {benchmark_name}")
    print(f"Repetition: {df.repetition.iloc[0]}")
    print(f"Test: {test}")
    print('p-value=%.3f\nstatistics=%.3f' %
          (df.p_value.iloc[0], df.statistics.iloc[0]))
    if p > 0.05:
        print('\nProbably the same distribution')
    else:
        print('Probably different distributions')
    print("")


def _print_results_post_hoc(df: pd.DataFrame, test):
    benchmark_name = load_benchmark_by_identifier(
        [int(df.benchmark_id.iloc[0])])[0]['name']
    print(f"Task: {df.task.iloc[0]}")
    print(f"Benchmark: {benchmark_name}")
    print(f"Repetition: {df.repetition.iloc[0]}")
    print(
        f"Test: {test}\n{tabulate.tabulate(df['result'].iloc[0],headers='keys',tablefmt='pretty')}\n"
    )


def print_results(results: pd.DataFrame, test):

    results.groupby(['repetition', 'tag', 'task', 'benchmark_id'
                     ]).apply(lambda _df: _print_results(_df, test))


def print_results_post_hoc(results: pd.DataFrame, test):

    results.groupby(['repetition', 'tag', 'task', 'benchmark_id'
                     ]).apply(lambda _df: _print_results_post_hoc(_df, test))
