
from nicegui import ui
import src.functions.register as register
from src.handler import solver_handler,benchmark_handler,config_handler,experiment_handler
from os import getcwd

from src.functions import plot,statistics,score,table_export
from src.functions import validation,plot_validation,validation_table_export
from src.functions import parametric_significance,parametric_post_hoc
from src.functions import non_parametric_significance,non_parametric_post_hoc
from src.functions import plot_significance,post_hoc_table_export,plot_post_hoc



def _build_solver_table_options():
    column_defs =  [
                    {'headerName': 'ID', 'field': 'id'},
                    {'headerName': 'Name', 'field': 'name'},
                    {'headerName': 'Version', 'field': 'version'},
                    {'headerName': 'Format', 'field': 'format'},
                    ]
    solvers = solver_handler.load_solver('all')
    row_data = [{'id': s['id'],'name': s['name'],'version': s['version'],'format': s['format']} for s in solvers]

    return { 'columnDefs': column_defs,'rowData':row_data,'rowSelection':'multiple',
    'rowMultiSelectWithClick': True,
    'autoHeight': True}

def _build_benchmark_table_options():
    column_defs =  [
                    {'headerName': 'ID', 'field': 'id'},
                    {'headerName': 'Name', 'field': 'name'},
                    {'headerName': 'Format', 'field': 'format'},
                    {'headerName': 'Additional Ext.', 'field': 'ext_additional'},
                    ]
    benchmarks = benchmark_handler.load_benchmark('all')
    row_data = [{'id': s['id'],'name': s['name'],'ext_additional': s['ext_additional'],'format': s['format']} for s in benchmarks]

    return { 'columnDefs': column_defs,'rowData':row_data,'rowSelection':'multiple',
    'rowMultiSelectWithClick': True,
    'autoHeight': True}

selected_solvers = set()
selected_benchmarks = set()

def handle_solver_selection_click(sender, msg):

    selected_solver_id = msg.data['id']
    if msg.selected:
        selected_solvers.add(selected_solver_id)
    else:
        selected_solvers.remove(selected_solver_id)

def handle_benchmark_selection_click(sender, msg):

    selected_benchmark_id = msg.data['id']
    if msg.selected:
        selected_benchmarks.add(selected_benchmark_id)
    else:
        selected_benchmarks.remove(selected_benchmark_id)










params = {  'name':None,
            'timeout': None,
            'repetitions':None,
            'task':None,
            'exclude_task':None,
            'solver':None,
            'benchmark':None,
            'save_to': None,
            'archive':None,
            'save_output': False,
            'copy_raws': False,
            'plot':None,
            'statistics': None,
            'score': None,
            'validation': None,
            'significance': None,
            'table_export': None }
card_index = 0
container = None
log=None

def set_parameter(key,value):
    params[key] = value

def show_side_menu():
    with ui.left_drawer().props('width=250').classes('bg-grey-3'):
        ui.button('Experiment Status',on_click=lambda e:ui.open(status_page,e.socket)).props('icon="hourglass_empty"').props('flat')
        ui.button('Solvers',on_click=lambda e:ui.open(solvers_page,e.socket)).props('icon="functions"').props('flat')
        ui.button('Benchmarks',on_click=lambda e:ui.open(benchmarks_page,e.socket)).props('icon="description"').props('flat')





def show_settings():
    with ui.card():
        #ui.badge('Test')
        with ui.card_section().classes('bg-grey-3'):
                ui.label('Probo2 Web').classes('text-h6')
                ui.label('Experiment Configuration').classes('text-subtitle2')
        with ui.expansion(text='General',icon='settings').classes('w-full').props('expand-separator'):
            #ui.label('General Settings').classes('font-bold').style('color: #6E93D6; font-size: 200%; font-weight: 300').props('header')

            name = ui.input('Experiment Name',on_change=lambda e: set_parameter('name',e.value))

            timeout = ui.number(label='Timeout in seconds', value=600,on_change=lambda e:set_parameter('timeout',e.value))
            set_parameter('timeout',timeout.value)

            repetitions = ui.number(label='Repeat Runs', value=1,on_change=lambda e:set_parameter('repetitions',e.value))
            set_parameter('repetitions',repetitions.value)

            task = ui.input('Tasks to run', placeholder='e.g. EE-PR, DC-PR or "supported"',on_change=lambda e:set_parameter('task',e.value.split(','))).props('autogrow clearable')

            exclude_task = ui.input('Exclude Tasks', placeholder='e.g. EE-,-PR',on_change=lambda e:set_parameter('exclude_task',e.value))
            set_parameter('task',task.value)
            set_parameter('exclude_task',exclude_task.value)

            with ui.expansion(text='Advanced').classes('w-full').props('expand-separator icon=settings'):
                with ui.row():
                    save_output = ui.checkbox('Save Output',value=False,on_change=lambda e:set_parameter('save_output',e.value))
                    copy_raws = ui.checkbox('Copy Raws',value=False,on_change=lambda e:set_parameter('copy_raws',e.value))

                archive = ui.select(['zip','None'],label='Archive Output',on_change=lambda e:set_parameter('archive',e.value)).classes('w-full')
                save_to = ui.input('Save Directory', placeholder=f'Current:{getcwd()}',on_change=lambda e:set_parameter('save_to',e.value))
                set_parameter('save_to', getcwd())
        with ui.expansion(text='Solver',icon='functions').classes('w-full').props('expand-separator caption="Input or select solvers"'):
            show_solver_card()

        with ui.expansion(text='Benchmarks',icon='description').classes('w-full').props('expand-separator caption="Input or select benchmarks"'):
            show_benchmark_card()
        with ui.expansion(text='Analysis',icon='equalizer').classes('w-full').props('expand-separator caption="Plots, Statistics, Validation"'):
            plot = ui.select([p.capitalize() for p in register.plot_dict.keys()] + ['all'],label='Plot Type').props('multiple')
            test = ui.label("")
            statistics =  ui.select([p.capitalize() for p in register.stat_dict.keys()] + ['all'],label='Statistics').props('multiple')
            score =  ui.select([p.capitalize() for p in register.score_functions_dict.keys()] + ['all'],label='Scores').props('multiple')
            table_export = ui.select(list(register.table_export_functions_dict.keys())+['all'],label='Tabel Export').props('multiple')

            with ui.expansion(text='Validation',icon='compare').classes('w-full').props('expand-separator caption="Mode, Plots, Tables"'):
                validation = {'mode': None, 'plot': None,'table_export':None,'references': None}
                val_mode =  ui.select([p.capitalize() for p in register.validation_functions_dict.keys()] + ['all'],label='Validation Mode').props('multiple')
                vaL_reference = ui.input('Reference Files', placeholder=f'Current:{getcwd()}',on_change=lambda e:set_value(validation,'references',e.value))
                val_plot =  ui.select([p.capitalize() for p in register.plot_validation_functions_dict.keys()] + ['all'],label='Validation Plot').props('multiple')
                val_table =  ui.select([p for p in register.validation_table_export_functions_dict.keys()] + ['all'],label='Validation Tables').props('multiple')


            with ui.expansion(text='Significance Tests',icon='exposure').classes('w-full').props('expand-separator caption="(Non-)Parametric, Post-Hoc,Plots"'):
                significance = {'parametric_test': None,
                                'parametric_post_hoc': None,
                                'non_parametric_test': None,
                                'non_parametric_post_hoc': None,
                                'p_adjust': None,
                                'plot': None,
                                'table_export': None}
                parametric_test =  ui.select(list(register.parametric_significance_functions_dict.keys()) + ['all'],label='Parametric Tests').props('multiple')
                parametric_post_hoc =  ui.select(list(register.parametric_post_hoc_functions_dict.keys()) + ['all'],label='Parametric Post-Hoc').props('multiple')
                non_parametric_test =  ui.select(list(register.non_parametric_significance_functions_dict.keys()) + ['all'],label='Non-Parametric Tests').props('multiple')
                non_parametric_post_hoc =  ui.select(list(register.non_parametric_post_hoc_functions_dict.keys()) + ['all'],label='Non-Parametric Post-Hoc').props('multiple')
                p_adhust = ui.select(['holm'],label='P-Adjust',on_change=lambda e:set_value(significance,'p_adjust',e.value))
                post_hoc_plot =  ui.select(list(register.plot_post_hoc_functions_dict.keys()) + ['all'],label='Post-Hoc Plots').props('multiple')
                post_hoc_table_export =  ui.select(list(register.post_hoc_table_export_functions_dict.keys()) + ['all'],label='Post-Hoc Tables').props('multiple')




        def _update_multiple_selection():

            plot_selections = _extract_values_from_view(plot.view)
            set_parameter('plot',plot_selections)

            stats_selections = _extract_values_from_view(statistics.view)
            set_parameter('statistics',stats_selections)

            score_selection = _extract_values_from_view(score.view)
            set_parameter('score',score_selection)

            table_export_selection = _extract_values_from_view(table_export.view)
            set_parameter('table_exprt',table_export_selection)


            val_mode_selection = _extract_values_from_view(val_mode.view)
            val_plot_selection = _extract_values_from_view(val_plot.view)
            val_table_selection = _extract_values_from_view(val_table.view)
            validation['mode'] = val_mode_selection
            validation['plot'] = val_plot_selection
            validation['table_export'] = val_table_selection
            set_parameter('validation',validation)

            parametric_test_selection = _extract_values_from_view(parametric_test.view)
            non_parametric_test_selection = _extract_values_from_view(non_parametric_test.view)
            parametric_post_hoc_selection = _extract_values_from_view(parametric_post_hoc.view)
            non_parametric_post_hoc_selection = _extract_values_from_view(non_parametric_post_hoc.view)
            post_hoc_plot_selection = _extract_values_from_view(post_hoc_plot.view)
            post_hoc_table_export_selection = _extract_values_from_view(post_hoc_table_export.view)

            significance['parametric_test'] = parametric_test_selection
            significance['non_parametric_test'] = non_parametric_test_selection
            significance['parametric_post_hoc'] = parametric_post_hoc_selection
            significance['non_parametric_post_hoc'] = non_parametric_post_hoc_selection
            significance['plot'] = post_hoc_plot_selection
            significance['table_export'] = post_hoc_table_export_selection

            set_parameter('significance',significance)

        with ui.row().classes('flex justify-center w-full'):
            cfg = config_handler.load_default_config()
            def prepare_run():
                _update_multiple_selection()
                cfg.merge_user_input(params)
                is_valid = cfg.check()
                if is_valid:
                    summary_dialog.open()
                else:
                    show_invalid_cfg_dialog()


            def start_experiment():
                summary_dialog.close()
                experiment_handler.run_pipeline(cfg)



            ui.button("Run Experiment",on_click=lambda: prepare_run()).props('icon=play_arrow color=positive')

            with ui.dialog() as summary_dialog, ui.card():
                show_experiment_summary(cfg)
                with ui.row().classes('flex justify-center w-full'):
                    ui.button(on_click=lambda: start_experiment()).props('round icon=play_arrow color=positive')
                    ui.button(on_click=lambda: summary_dialog.close()).props('round icon=close color=negative')


def show_invalid_cfg_dialog():

    ui.notify('Bad Config found!')


def show_experiment_summary(cfg: config_handler.Config):
    ui.label('Experiment Summary').classes('text-h6')
    with ui.row():

        ui.label('Name: ')
        ui.label('').bind_text_from(cfg,'name')
    with ui.row():

        ui.label('Task: ')
        ui.label('').bind_text_from(cfg,'task')
    with ui.row():

        ui.label('Timeout: ')
        ui.label('').bind_text_from(cfg,'timeout')
    with ui.row():

        ui.label('Solver: ')
        ui.label('').bind_text_from(cfg,'solver')
    with ui.row():

        ui.label('Benchmark: ')
        ui.label('').bind_text_from(cfg,'benchmark')
    with ui.row():

        ui.label('Reps: ')
        ui.label('').bind_text_from(cfg,'repetitions')






def _extract_values_from_view(view):
    if view.value is not None:
        return [ v['label'].lower() for v in view.value ]
    else:
        return None


def set_value(_dict,key,value):
    if key in _dict.keys():
         _dict[key] = value






def show_solver_card():

    async def show_solver_selection():
                result = await dialog
                if result is None:
                    ui.notify(f'Canceled')
                else:
                    ui.notify(f'Solver {",".join(map(str,selected_solvers))} selected')
                    set_parameter('solver', list(selected_solvers))
                    solver_list.set_text(",".join(map(str,selected_solvers)))

    def remove_solvers():
        selected_solvers.clear()
        set_parameter('solver',selected_solvers)
        solver_list.set_text("")

    table_options = _build_solver_table_options()


    #ui.label('Solver Settings').classes('font-bold').style('color: #6E93D6; font-size: 200%; font-weight: 300')

    with ui.column():#.classes('flex justify-center w-full'):
        with ui.row().classes('flex justify-center w-full'):
            solver = ui.input('Solvers to run ', placeholder='Name or ID of solver',on_change=lambda e:set_parameter('solver',e.value.split(','))).props('autogrow')


        with ui.row().classes('flex justify-center w-full'):
            ui.button('Select', on_click=show_solver_selection)
            ui.button('Clear',on_click=remove_solvers)
        with ui.row().classes('flex justify-center w-full'):
            ui.label('Selected:')
            solver_list = ui.label()
        with ui.dialog() as dialog:
            with ui.card().classes('flex justify-center w-1/4').style('overflow: hidden'):
                table = ui.table(table_options).classes('max-h-80')
                #table.view.on('cellClicked', handle_solver_selection_click)
                table.call_api_method('cellClicked',handle_solver_selection_click)
                with ui.row().classes('flex justify-center w-full'):
                    ui.button(on_click=lambda: dialog.submit(selected_solvers)).props('round icon=done color=positive')
                    ui.button(on_click=lambda: dialog.submit(None)).props('round icon=close color=negative')

def show_benchmark_card():

    async def show_benchmark_selection():
                result = await dialog
                if result is None:
                    ui.notify(f'Canceled')
                else:
                    ui.notify(f'Benchmark {",".join(map(str,selected_benchmarks))} selected')
                    set_parameter('benchmark', list(selected_benchmarks))
                    benchmark_list.set_text(",".join(map(str,selected_benchmarks)))

    def remove_benchmarks():
        selected_benchmarks.clear()
        set_parameter('benchmark',selected_benchmarks)
        benchmark_list.set_text("")

    table_options = _build_benchmark_table_options()


    #ui.label('Benchmark Settings').classes('font-bold').style('color: #6E93D6; font-size: 200%; font-weight: 300')

    with ui.column().classes('flex justify-center w-full'):
        with ui.row().classes('flex justify-center w-full'):
            solver = ui.input('Benchmarks to run ', placeholder='Name or ID of benchmark',on_change=lambda e:set_parameter('benchmark',e.value.split(',')))


        with ui.row().classes('flex justify-center w-full'):
            ui.button('Select', on_click=show_benchmark_selection)
            ui.button('Clear',on_click=remove_benchmarks)
        with ui.row().classes('flex justify-center w-full'):
            ui.label('Selected:')
            benchmark_list = ui.label()
        with ui.dialog() as dialog:
            with ui.card().classes('flex justify-center w-1/4').style('overflow: hidden'):
                table = ui.table(table_options).classes('max-h-80')
                #table.view.on('cellClicked', handle_benchmark_selection_click)
                table.call_api_method('cellClicked',handle_benchmark_selection_click)
                with ui.row().classes('flex justify-center w-full'):
                    ui.button(on_click=lambda: dialog.submit(selected_benchmarks)).props('round icon=done color=positive')
                    ui.button(on_click=lambda: dialog.submit(None)).props('round icon=close color=negative')




#card_dict = {0: show_settings,1:show_solver_card,2:show_benchmark_card}

# def decrement_card():
#     global card_index
#     card_index -= 1
#     if card_index in card_dict.keys():
#         card_dict[card_index]()
#     else:
#         card_index += 1

# def increment_card():
#     global card_index
#     card_index += 1
#     if card_index in card_dict.keys():
#         card_dict[card_index]()
#     else:
#         card_index -= 1
def build_interface():
    with ui.row().classes('flex justify-center w-full mt-20'):
        show_settings()


@ui.page('/')
def main_page() -> None:
    show_side_menu()
    build_interface()
    #table_in_dialog()


import json
from src.utils import definitions

@ui.page('/status')
def status_page() -> None:
    with open(str(definitions.STATUS_FILE_DIR)) as status_json_file:
        status_data = json.load(status_json_file)
    with ui.row().classes('flex justify-center w-full'):
        with ui.card():
            with ui.card_section():
                ui.label('Experiment Progress').classes('text-h6')
                with ui.row().classes('flex justify-center w-full'):
                    ui.label(f'{status_data["name"]}').classes('text-subtitle2')
            with ui.row():
                ui.label(f'Tasks finished : {status_data["finished_tasks"]}/{status_data["total_tasks"]}')
                    #ui.circular_progress(value=status_data['finished_tasks'],min=0,max=status_data['total_tasks'])
            for task in status_data['tasks'].keys():
                current_task = status_data['tasks'][task]
                with ui.expansion(text=f'{task}'):
                    for solver in current_task['solvers']:
                        current_solver = current_task['solvers'][solver]
                        ui.label(f'{current_solver["name"]}: {current_solver["solved"]}/{current_solver["total"]} ')

@ui.page('/solvers/add')
def add_solver_page():
    from src.handler import solver_handler
    add_solver_options = solver_handler.AddSolverOptions('','','',True,None,None,True,False)
    show_solvers_side_menu()

    with ui.row().classes('flex justify-center w-full'):
        with ui.card():
            #ui.badge('Test')
            with ui.card_section():
                    ui.label('Probo2 Web').classes('text-h6')
                    ui.label('Add Solver').classes('text-subtitle2')
            name = ui.input('Name',placeholder='MyAwesomeSolver').bind_value(add_solver_options,'name')
            path = ui.input('Path',placeholder='path/to/solver').bind_value(add_solver_options,'path')
            version = ui.input('Version',placeholder='3.141').bind_value(add_solver_options,'version')

            fetch = ui.checkbox('Fetch Task, Format').bind_value(add_solver_options,'fetch')
            no_check = ui.checkbox('No Check',value=False).bind_value(add_solver_options,'no_check')
            with ui.row().classes('flex justify-center w-full'):
                ui.button('Add',on_click=lambda: _add_solver(add_solver_options) ).props('icon="add" color=positive')

import yaml
def _add_solver(options: solver_handler.AddSolverOptions):
    new_solver = solver_handler._create_solver_obj(options)

    with ui.dialog() as dialog, ui.card():
        ui.label(yaml.safe_dump(new_solver.__dict__))
        with ui.row():
            ui.button('Yes', on_click=lambda: _check_solver_interface(dialog,options,new_solver))
            ui.button('No', on_click=lambda: dialog.close())
        dialog.open()

def _check_solver_interface(d:ui.dialog,options: solver_handler.AddSolverOptions,new_solver):

    if not options.no_check:
        is_working = solver_handler.check_interface(new_solver)
    else:
        is_working = True

    if not is_working:
        ui.notify('Something went wrong when testing the interface!')
        d.close()
        return
    id = solver_handler._add_solver_to_database(new_solver)
    ui.notify(f'Solver added with id: {id}')
    d.close()

solver_to_delete = set()
def handle_solver_to_delete_selection(sender, msg):
    solver_to_delete_id = msg.data['id']
    if msg.selected:
        solver_to_delete.add(solver_to_delete_id)
    else:
        solver_to_delete.remove(solver_to_delete_id)

@ui.page('/solvers/delete')
def delete_solver_page():
    show_solvers_side_menu()
    table_options = _build_solver_table_options()
    def delete_solvers():
        if solver_to_delete:
            for _solver in solver_to_delete:
                solver_handler.delete_solver(_solver)
        table_options = _build_solver_table_options()
        table.options = table_options
        table.update()
    with ui.row().classes('flex justify-center w-full'):

        with ui.card().classes('flex justify-center w-1/4').style('overflow: hidden'):
            table = ui.table(table_options).classes('max-h-80')
            table.view.on('cellClicked', handle_solver_to_delete_selection)
            ui.button('Delete',on_click=delete_solvers)




@ui.page('/solvers/database')
def database_solver_page():
    show_solvers_side_menu()
    table_options = _build_solver_table_options()
    with ui.row().classes('flex justify-center w-full'):
        with ui.card().classes('flex justify-center w-1/3').style('overflow: hidden'):
            ui.table(table_options)




def show_solvers_side_menu()-> None:
    with ui.left_drawer().props('width=200').classes('bg-grey-3'):
        with ui.column():
            ui.button('Add',on_click=lambda e:ui.open(add_solver_page,e.socket)).props('icon="add"').props('flat')
            ui.button('Delete',on_click=lambda e:ui.open(delete_solver_page,e.socket)).props('icon="delete"').props('flat')
            ui.button('Show',on_click=lambda e:ui.open(database_solver_page,e.socket)).props('icon="description"').props('flat')




@ui.page('/solvers')
def solvers_page() -> None:
    #load solvers.md
    with open(definitions.SOLVERS_MD,'r') as md:
        md_content = md.read()
    show_solvers_side_menu()
    with ui.row().classes('flex justify-center w-full'):
        ui.markdown(f'''{md_content}''')


@ui.page('/benchmarks')
def benchmarks_page() -> None:
    ui.label('Welcome')


ui.run(title='probo2')





