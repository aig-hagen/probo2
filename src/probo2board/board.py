import pandas as pd
import panel as pn
import hvplot.pandas
import holoviews as hv
import numpy as np
import hvplot
from src.functions import register,statistics
from functools import reduce

hv.extension('bokeh', 'plotly')

pn.extension('tabulator', sizing_mode="stretch_width")
import src.probo2board.pipelines as pipelines
from src.utils import benchmark_handler

STATS = ['sum','mean','solved','errors','timeouts','coverage']

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
    
    # ========== Tables ==========
    table_df_pipline = pipelines.get_pipeline_table(df,reps,tasks,benchmarks,solvers)
    table = table_df_pipline.pipe(pn.widgets.Tabulator, pagination='remote', page_size=20)
    stats_results = []
    for stat in STATS:
        _res = register.stat_dict[stat](df,avg_reps=False)
        stats_results.append(_res[0]) # DataFrame is on position 0, 
    
    df_stats_merged = reduce(lambda  left,right: pd.merge(left,right,how='inner'), stats_results)
    df_stats_merged['solver_full_name'] = df_stats_merged.solver_name +'_'+ df_stats_merged.solver_version
    df_stats_merged = df_stats_merged.drop(['tag','solver_name','solver_version'],axis=1)
    stats_table_df_pipline = pipelines.get_pipeline_table(df_stats_merged[df_stats_merged.columns[::-1]],reps,tasks,benchmarks,solvers)
    stats_table = stats_table_df_pipline.pipe(pn.widgets.Tabulator, pagination='remote', page_size=20)


    # ========== PLOTS ==========
    count_plot_df = prep_data_count_plot(df)
    count_pipeline =  pipelines.get_pipeline_with_rep_count_plots(count_plot_df,reps,tasks,benchmarks,solvers)
    count_plot = count_pipeline.hvplot(rot=45,stacked=True,y=['Solved','Timed out','Aborted'],x='solver_full_name',responsive=True,height=400,kind='bar',title='Count Plot')

    ranked_df = (df[(df.timed_out == False) &
         (df.exit_with_error == False) ].groupby(['repetition','task','benchmark_id','solver_id'],group_keys=False)
                                  .apply(lambda df: df.assign(rank=df['runtime'].rank(method='dense',ascending=True))).sort_values(by=['rank']))

    cactus_pipeline = pipelines.get_pipeline_with_rep_cactus(ranked_df,reps,tasks,benchmarks,solvers)
    # Create a cactus plot for each pipeline (interactive dataframe)
    line_plot = cactus_pipeline.hvplot(x='rank',y='runtime',by=['solver_full_name','task','benchmark_name','repetition'], line_width=3,responsive=True, height=400,title='Cactus Plot').opts(show_legend=False,fontscale=0.7)
    scatter_plot = cactus_pipeline.hvplot(x='rank',y='runtime',by=['solver_full_name','task','benchmark_name','repetition'],responsive=True, height=400,kind='scatter').opts(show_legend=False)
    cactus_plot = (line_plot * scatter_plot)

    # ========== HEADER TEXT =========
    settings_html = pn.pane.HTML("""
    <h1>Settings</h1>""",style={'color': '#0E559C'})
    rep_html = pn.pane.HTML("""
    <h3>Repetitions</h3>""",style={'color': '#0E559C'})
    tasks_html = pn.pane.HTML("""
    <h3>Tasks</h3>""",style={'color': '#0E559C'})
    benchmarks_html = pn.pane.HTML("""
    <h3>Benchmarks</h3>""",style={'color': '#0E559C'})
    solver_html = pn.pane.HTML("""
    <h3>Solvers</h3>""",style={'color': '#0E559C'})

    summary_html = pn.pane.HTML("""
    <h2>Summary</h2>""")
    summary_widget = generate_summary(df)

    #========== BACKEND TOGGLES =========
    backend_toggle = pn.widgets.RadioButtonGroup(options=['bokeh', 'plotly'],value='bokeh') # matlibplot does not work
    watcher = backend_toggle.param.watch(callback, ['options', 'value'], onlychanged=False)



    template = pn.template.FastListTemplate(
    title=f'Probo2Board-{df.iloc[0].tag}', 
    sidebar=[backend_toggle,settings_html,rep_html,reps,tasks_html,tasks,benchmarks_html,benchmarks,solver_html,solvers],
    main=[summary_html,summary_widget,pn.pane.Markdown('## Plots'),cactus_plot.panel(),count_plot.panel(),pn.pane.Markdown('## Statistics'),stats_table.panel(),pn.pane.Markdown('## Raw'),table.panel()],
    accent_base_color="#7BA0C8",
    header_background="#0E559C",
    logo='https://www.fernuni-hagen.de/aig/images/aig_logo.png',
    sidebar_width=200,
    )
    template.show()



def generate_summary(df):
    tag = df.iloc[0].tag
    tasks = ",".join(list(df.task.unique()))
    benchmark_id_map = df[['benchmark_id','benchmark_name']].drop_duplicates()
    benchmark_id_map = dict(zip(benchmark_id_map.benchmark_id, benchmark_id_map.benchmark_name))
    benchmarks_text =  ','.join([f'{n}(ID:{i})' for i,n in benchmark_id_map.items()])
    
    solver_id_map = df[['solver_id','solver_full_name']].drop_duplicates()
    solver_id_map = dict(zip(solver_id_map.solver_id, solver_id_map.solver_full_name))
    solvers_text =  ','.join([f'{n}(ID:{i})' for i,n in solver_id_map.items()])
    
    timeout = df.iloc[0].cut_off
    reps_text = ','.join(map(str,list(df.repetition.unique())))
    
    summary_text=f'Expepriment: {tag}\n\nTasks: {tasks}\n\nBenchmarks: {benchmarks_text}\n\nSolvers: {solvers_text}\n\nTimeout: {timeout}\n\nRepetitions: {reps_text}'
    
    return pn.pane.Str(summary_text,style={'color': '#818589'})

def prep_data_count_plot(df):
    conds = [((df.timed_out == False) & (df.exit_with_error == False)),df.timed_out == True,df.exit_with_error == True]
    choices = ['Solved','Timed out','Aborted']
    df['Status'] = np.select(conds,choices)
    value_counts = df.groupby(['repetition','task','benchmark_id','solver_id'],group_keys=True).apply(lambda _df: _df.value_counts(['Status'])).unstack(level='Status').reset_index()
   
    benchmarks = benchmark_handler.load_benchmark(list(df.benchmark_id.unique()))
    
    benchmark_id_name_map = { b['id']:b['name'] for b in benchmarks}
  

    df['benchmark_name'] = df['benchmark_id'].map(benchmark_id_name_map)
    solver_id_map = df[['solver_id','solver_full_name']].drop_duplicates()
    solver_id_map = dict(zip(solver_id_map.solver_id, solver_id_map.solver_full_name))
    

    benchmark_id_map = df[['benchmark_id','benchmark_name']].drop_duplicates()
    benchmark_id_map = dict(zip(benchmark_id_map.benchmark_id, benchmark_id_map.benchmark_name))
    value_counts['solver_full_name'] = value_counts.solver_id.map(solver_id_map)
    value_counts['benchmark_name'] = value_counts.benchmark_id.map(benchmark_id_map)
    if 'Solved' not in value_counts.columns:
        value_counts['Solved'] = 0
    if 'Timed out' not in value_counts.columns:
        value_counts['Timed out'] = 0
    if 'Aborted' not in value_counts.columns:
        value_counts['Aborted'] = 0

    return value_counts.fillna(0)
    

def callback(*events):
   
    for event in events:
        obj = event.obj
        hv.output(backend=obj.value)

if __name__ == '__main__':
    df = pd.read_csv("/home/jklein/dev/probo2/src/results/probo2_demo/raw.csv")
    launch(df)