import src.functions.register as register
import pandas as pd
import numpy as np
import src.handler.config_handler as config_handler

import src.functions.register as register
import pandas as pd
import numpy as np
import src.handler.config_handler as config_handler
import scikit_posthocs as sp

def test(df: pd.DataFrame, cfg: config_handler.Config):


    significance_cfg = cfg.significance
    if significance_cfg['non_parametric_post_hoc'] =='all' or 'all' in significance_cfg['non_parametric_post_hoc']:
            significance_cfg['non_parametric_post_hoc'] = register.non_parametric_post_hoc_functions_dict.keys()
    results = {}
    for test in significance_cfg['non_parametric_post_hoc']:
        if test in register.non_parametric_post_hoc_functions_dict.keys():
            _res = register.non_parametric_post_hoc_functions_dict[test](df, cfg)

            results[test] = _res


    # for test, result_df in results.items():
    #     for index, row in result_df.iterrows():
    #         print(row.result)


    return results


def _post_hoc_conover(df: pd.DataFrame, p_adjust):
    result = sp.posthoc_conover(df, val_col='runtime', group_col='solver_id', p_adjust=p_adjust)
    id_name_map = dict(zip(df.solver_id,df.solver_name))
    result.columns = result.columns.map(id_name_map)
    result.index = result.index.map(id_name_map)
    return pd.Series({'result':result})


def post_hoc_conover(df: pd.DataFrame, cfg: config_handler.Config):
    return df.groupby(['repetition','tag','task','benchmark_id'], as_index=False).apply(lambda _df: _post_hoc_conover(_df, cfg.significance['p_adjust']))


def _post_hoc_mannwhitney(df: pd.DataFrame):
    result = sp.posthoc_mannwhitney(df, val_col='runtime', group_col='solver_id')
    id_name_map = dict(zip(df.solver_id,df.solver_name))
    result.columns = result.columns.map(id_name_map)
    result.index = result.index.map(id_name_map)
    return pd.Series({'result':result})


def post_hoc_mannwhitney(df: pd.DataFrame, cfg: config_handler.Config):
    return df.groupby(['repetition','tag','task','benchmark_id'], as_index=False).apply(lambda _df: _post_hoc_mannwhitney(_df))



def post_hoc_dscf(df: pd.DataFrame, cfg: config_handler.Config):
    return df.groupby(['repetition','tag','task','benchmark_id'], as_index=False).apply(lambda _df: _post_hoc_dscf(_df))
def _post_hoc_dscf(df: pd.DataFrame):
    result = sp.posthoc_dscf(df, val_col='runtime', group_col='solver_id')
    id_name_map = dict(zip(df.solver_id,df.solver_name))
    result.columns = result.columns.map(id_name_map)
    result.index = result.index.map(id_name_map)
    return pd.Series({'result':result})

def post_hoc_nemenyi(df: pd.DataFrame, cfg: config_handler.Config):
    return df.groupby(['repetition','tag','task','benchmark_id'], as_index=False).apply(lambda _df: _post_hoc_nemenyi(_df))

def _post_hoc_nemenyi(df: pd.DataFrame):
    result = sp.posthoc_nemenyi(df, val_col='runtime', group_col='solver_id')
    id_name_map = dict(zip(df.solver_id,df.solver_name))
    result.columns = result.columns.map(id_name_map)
    result.index = result.index.map(id_name_map)
    return pd.Series({'result':result})


def post_hoc_dunn(df: pd.DataFrame, cfg: config_handler.Config):
    return df.groupby(['repetition','tag','task','benchmark_id'], as_index=False).apply(lambda _df: _post_hoc_dunn(_df,cfg.significance['p_adjust']))

def _post_hoc_dunn(df: pd.DataFrame,p_adjust):
    result = sp.posthoc_dunn(df, val_col='runtime', group_col='solver_id',p_adjust=p_adjust)
    id_name_map = dict(zip(df.solver_id,df.solver_name))
    result.columns = result.columns.map(id_name_map)
    result.index = result.index.map(id_name_map)
    return pd.Series({'result':result})

def post_hoc_npm_test(df: pd.DataFrame, cfg: config_handler.Config):
    return df.groupby(['repetition','tag','task','benchmark_id'], as_index=False).apply(lambda _df: _post_hoc_npm_test(_df))

def _post_hoc_npm_test(df: pd.DataFrame):
    result = sp.posthoc_npm_test(df, val_col='runtime', group_col='solver_id')
    id_name_map = dict(zip(df.solver_id,df.solver_name))
    result.columns = result.columns.map(id_name_map)
    result.index = result.index.map(id_name_map)
    return pd.Series({'result':result})

def post_hoc_vanwaerden(df: pd.DataFrame, cfg: config_handler.Config):
    return df.groupby(['repetition','tag','task','benchmark_id'], as_index=False).apply(lambda _df: _post_hoc_vanwaerden(_df,cfg.significance['p_adjust']))

def _post_hoc_vanwaerden(df: pd.DataFrame,p_adjust):
    result = sp.posthoc_vanwaerden(df, val_col='runtime', group_col='solver_id',p_adjust=p_adjust)
    id_name_map = dict(zip(df.solver_id,df.solver_name))
    result.columns = result.columns.map(id_name_map)
    result.index = result.index.map(id_name_map)
    return pd.Series({'result':result})

def post_hoc_wilcoxon(df: pd.DataFrame, cfg: config_handler.Config):
    return df.groupby(['repetition','tag','task','benchmark_id'], as_index=False).apply(lambda _df: _post_hoc_wilcoxon(_df,cfg.significance['p_adjust']))

def _post_hoc_wilcoxon(df: pd.DataFrame,p_adjust):
    result = sp.posthoc_wilcoxon(df, val_col='runtime', group_col='solver_id',p_adjust=p_adjust)
    id_name_map = dict(zip(df.solver_id,df.solver_name))
    result.columns = result.columns.map(id_name_map)
    result.index = result.index.map(id_name_map)
    return pd.Series({'result':result})







register.non_parametric_post_hoc_functions_register('conover',post_hoc_conover)
register.non_parametric_post_hoc_functions_register('mannwhitney',post_hoc_mannwhitney)
register.non_parametric_post_hoc_functions_register('dscf',post_hoc_dscf)
register.non_parametric_post_hoc_functions_register('nemenyi',post_hoc_nemenyi)
register.non_parametric_post_hoc_functions_register('dunn',post_hoc_dunn)
register.non_parametric_post_hoc_functions_register('npm_test',post_hoc_npm_test)
register.non_parametric_post_hoc_functions_register('vanwaerden',post_hoc_vanwaerden)
register.non_parametric_post_hoc_functions_register('wilcoxon',post_hoc_wilcoxon)
