import src.functions.register as register
import pandas as pd
import numpy as np
import src.utils.config_handler as config_handler

def test(df: pd.DataFrame, cfg: config_handler.Config):
    significance_cfg = cfg.significance
    if significance_cfg['test'] =='all' or 'all' in significance_cfg['test']:
            significance_cfg['test'] = register.significance_functions_dict.keys()
    results = {}
    for test in significance_cfg['test']:
        if test in register.significance_functions_register.keys():
            _res = register.significance_functions_dict[test](df, cfg)
            results[test] = _res
    return results