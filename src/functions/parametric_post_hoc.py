import src.functions.register as register
import pandas as pd
import numpy as np
import src.utils.config_handler as config_handler
import scikit_posthocs as sp

def test(df: pd.DataFrame, cfg: config_handler.Config):
    significance_cfg = cfg.significance
    if significance_cfg['parametric_post_hoc'] =='all' or 'all' in significance_cfg['parametric_post_hoc']:
            significance_cfg['parametric_post_hoc'] = register.parametric_post_hoc_functions_dict.keys()
    results = {}
    for test in significance_cfg['parametric_post_hoc']:
        if test in register.parametric_post_hoc_functions_dict.keys():
            _res = register.parametric_post_hoc_functions_dict[test](df, cfg)
            results[test] = _res
    return results


def _post_hoc_ttest(df: pd.DataFrame, p_adjust):
    result = sp.posthoc_ttest(df, val_col='runtime', group_col='solver_id', p_adjust=p_adjust)
    id_name_map = dict(zip(df.solver_id,df.solver_name))
    result.columns = result.columns.map(id_name_map)
    result.index = result.index.map(id_name_map)
    return pd.Series({'result':result})


def post_hoc_ttest(df: pd.DataFrame, cfg: config_handler.Config):
    return df.groupby(['repetition','tag','task','benchmark_id'], as_index=False).apply(lambda _df: _post_hoc_ttest(_df, cfg.significance['p_adjust']))

def _post_hoc_scheffe(df: pd.DataFrame):
    result = sp.posthoc_scheffe(df, val_col='runtime', group_col='solver_id')
    id_name_map = dict(zip(df.solver_id,df.solver_name))
    result.columns = result.columns.map(id_name_map)
    result.index = result.index.map(id_name_map)
    return pd.Series({'result':result})


def post_hoc_scheffe(df: pd.DataFrame, cfg: config_handler.Config):
    return df.groupby(['repetition','tag','task','benchmark_id'], as_index=False).apply(lambda _df: _post_hoc_scheffe(_df))



def post_hoc_tamhane(df: pd.DataFrame, cfg: config_handler.Config):
    return df.groupby(['repetition','tag','task','benchmark_id'], as_index=False).apply(lambda _df: _post_hoc_tamhane(_df))
def _post_hoc_tamhane(df: pd.DataFrame):
    result = sp.posthoc_tamhane(df, val_col='runtime', group_col='solver_id')
    id_name_map = dict(zip(df.solver_id,df.solver_name))
    result.columns = result.columns.map(id_name_map)
    result.index = result.index.map(id_name_map)
    return pd.Series({'result':result})

def post_hoc_tukey(df: pd.DataFrame, cfg: config_handler.Config):
    return df.groupby(['repetition','tag','task','benchmark_id'], as_index=False).apply(lambda _df: _post_hoc_tukey(_df))

def _post_hoc_tukey(df: pd.DataFrame):
    result = sp.posthoc_tukey(df, val_col='runtime', group_col='solver_id')
    id_name_map = dict(zip(df.solver_id,df.solver_name))
    result.columns = result.columns.map(id_name_map)
    result.index = result.index.map(id_name_map)
    return pd.Series({'result':result})




register.parametric_post_hoc_functions_register('ttest',post_hoc_ttest)
register.parametric_post_hoc_functions_register('scheffe',post_hoc_scheffe)
register.parametric_post_hoc_functions_register('tamhane',post_hoc_tamhane)
register.parametric_post_hoc_functions_register('tukey',post_hoc_tukey)
