import pandas as pd
import panel as pn
import hvplot.pandas
import holoviews as hv
import numpy as np
import hvplot

hv.extension('bokeh', 'plotly', 'matplotlib')

pn.extension('tabulator', sizing_mode="stretch_width")
import src.probo2board.pipelines as pipelines


def launch(df: pd.DataFrame) -> None:

    df['solver_full_name'] = df.solver_name +'_'+ df.solver_version
    # Define Options

    solver_options = sorted(list(df.solver_full_name.unique()))
    rep_options = sorted(list(df.repetition.unique()))
    tasks_options = sorted(list(df.task.unique()))
    benchmark_options = sorted(list(df.benchmark_name.unique()))

    # Init checkboxes
    reps = pn.widgets.CheckBoxGroup(name='Repetition',
                                    options=rep_options,
                                    value=rep_options,
                                    sizing_mode='scale_width')
    solvers = pn.widgets.CheckBoxGroup(name='Solver',
                                       options=solver_options,
                                       value=solver_options,
                                       sizing_mode='scale_width')
    tasks = pn.widgets.CheckBoxGroup(name='Tasks',
                                     options=tasks_options,
                                     value=tasks_options,
                                     sizing_mode='scale_width')
    benchmarks = pn.widgets.CheckBoxGroup(name='Benchmarks',
                                          options=benchmark_options,
                                          value=benchmark_options,
                                          sizing_mode='scale_width')
    
    table_df_pipeline = pipelines.get_pipeline_table(df,reps,tasks,benchmarks,solvers)
