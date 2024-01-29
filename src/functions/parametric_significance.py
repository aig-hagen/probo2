from typing_extensions import runtime
import src.functions.register as register
import pandas as pd
import numpy as np
import src.handler.config_handler as config_handler
import scipy.stats as scis

def test(df: pd.DataFrame, cfg: config_handler.Config):
    significance_cfg = cfg.significance
    if significance_cfg['parametric_test'] =='all' or 'all' in significance_cfg['parametric_test']:
            significance_cfg['parametric_test'] = register.parametric_significance_functions_dict.keys()
    results = {}
    for test in significance_cfg['parametric_test']:
        if test in register.parametric_significance_functions_dict.keys():
            _res = register.parametric_significance_functions_dict[test](df, cfg)
            results[test] = _res
    return results


def _anova(df: pd.DataFrame):
    runtimes = [ df[df.solver_id == solver].runtime.values for solver in df.solver_id.unique() ]

    stat, p = scis.f_oneway(*runtimes)
    return pd.Series({'p_value': p, 'statistics': stat})



def anova(df: pd.DataFrame, cfg: config_handler.Config):

    return df.groupby(['repetition','tag','task','benchmark_id'], as_index=False).apply(lambda _df: _anova(_df))


register.parametric_significance_functions_register('anova',anova)