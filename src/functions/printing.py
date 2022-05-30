import pandas as pd
import src.functions.register as register
from tabulate import tabulate
def default_print(df: pd.DataFrame, grouping):
    df.groupby(grouping).apply(lambda _df: print(tabulate(_df,headers='keys',tablefmt='fancy_grid',showindex=False)))



register.print_functions_register('default',default_print)


