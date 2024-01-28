import src.functions.register as register
import pandas as pd
import numpy as np
import src.utils.config_handler as config_handler
from scipy.stats import mannwhitneyu, kruskal


def test(df: pd.DataFrame, cfg: config_handler.Config):
    significance_cfg = cfg.significance
    if significance_cfg[
            'non_parametric_test'] == 'all' or 'all' in significance_cfg[
                'non_parametric_test']:
        significance_cfg[
            'non_parametric_test'] = register.non_parametric_significance_functions_dict.keys(
            )
    results = {}
    for test in significance_cfg['non_parametric_test']:
        if test in register.non_parametric_significance_functions_dict.keys():
            _res = register.non_parametric_significance_functions_dict[test](
                df, cfg)
            results[test] = _res
    return results


def _mannwhitneyu(df: pd.DataFrame):
    runtimes = [
        df[df.solver_id == solver].runtime.values
        for solver in df.solver_id.unique()
    ]

    stat, p = mannwhitneyu(*runtimes)
    return pd.Series({'p_value': p, 'f_statistics': stat})


def mannwhitneyu_test(df: pd.DataFrame, cfg: config_handler.Config):

    return df.groupby(['repetition', 'tag', 'task', 'benchmark_id'],
                      as_index=False).apply(lambda _df: _mannwhitneyu(_df))


def _kruskal(df: pd.DataFrame):
    runtimes = [
        df[df.solver_id == solver].runtime.values
        for solver in df.solver_id.unique()
    ]

    stat, p = kruskal(*runtimes)
    return pd.Series({'p_value': p, 'statistics': stat})


def print_result(test, stat, p):
    print(test.capitalize())
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')


def kruskal_test(df: pd.DataFrame, cfg: config_handler.Config):

    return df.groupby(['repetition', 'tag', 'task', 'benchmark_id'],
                      as_index=False).apply(lambda _df: _kruskal(_df))


register.non_parametric_significance_functions_register(
    'mannwhitneyu', mannwhitneyu_test)
register.non_parametric_significance_functions_register(
    'kruskal', kruskal_test)
