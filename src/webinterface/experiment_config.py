from nicegui import ui

selected_solvers = set()

table_options = {
    'columnDefs': [
        {'headerName': 'ID', 'field': 'id'},
        {'headerName': 'Name', 'field': 'name'},
        {'headerName': 'Version', 'field': 'version'},
        {'headerName': 'Format', 'field': 'format'},
            ],
    'rowData': [
        {'name': 'Mu-toksia', 'id': 18,'version': 1.0,'format': ['apx','tgf']},
        {'name': 'FUDGE', 'id': 12,'version': 1.3,'format': ['apx','tgf']},
        {'name': 'Pyglaf', 'id': 1,'version': 1.1,'format': ['apx','tgf']},
     
    ],
    'rowSelection':'multiple',
    'rowMultiSelectWithClick': True,
    'autoHeight': True
}

def handle_solver_selection_click(sender, msg):
    
    selected_solver_id = msg.data['id']
    if msg.selected:
        print(f'Solver selected: {selected_solver_id}')
        selected_solvers.add(selected_solver_id)
    else:
        print(f'Solver deselected: {selected_solver_id}')
        selected_solvers.remove(selected_solver_id)

def remove_solvers():
    selected_solvers.clear()

def build_general_settings() -> None:
     with ui.card().classes('flex justify-center'):
            with ui.row().classes('flex justify-center w-full'):
                name = ui.input('Experiment Name')
                
                    # task = ui.input('Tasks', placeholder='e.g. EE-PR, DC-PR')
                    # v = ui.checkbox('visible', value=True)
                    # with ui.column().bind_visibility_from(v, 'value'):
                    #     ui. checkbox('supported', value=True)
                    #     task = ui.input('Exclude Tasks', placeholder='e.g. EE-,-PR')
                    
                    
                task = ui.input('Tasks to run', placeholder='e.g. EE-PR, DC-PR or "supported"')
                exlude_task = ui.input('Exclude Tasks', placeholder='e.g. EE-,-PR')
                timeout = ui.number(label='Timeout in seconds', value=600)
                repetitions = ui.number(label='Repeat Runs', value=1)

            
            async def show_solver_selection():
                result = await dialog
                print(result)
                if result is None:
                    ui.notify(f'Canceled')
                else:
                    ui.notify(f'Solver {",".join(map(str,selected_solvers))} selected')
                    solver_list.set_text(f'Solver:{",".join(map(str,selected_solvers))}')
            with ui.expansion('Solver Settings!', icon='work').classes('w-1/2'):
                with ui.card():
                    with ui.column().classes('flex justify-center w-full'):
                        solver = ui.input('Solvers to run ', placeholder='Name or ID of solver')
                        ui.button('Select Solver', on_click=show_solver_selection)
                        ui.button('Remove Solver',on_click=remove_solvers)
                    solver_list = ui.label()
                    with ui.dialog() as dialog:
                        with ui.card().classes('flex justify-center w-1/4').style('overflow: hidden'):
                            table = ui.table(table_options).classes('max-h-80')
                            table.view.on('cellClicked', handle_solver_selection_click)
                            with ui.row().classes('flex justify-center w-full'):
                                ui.button(on_click=lambda: dialog.submit(selected_solvers)).props('round icon=done color=positive')
                                ui.button(on_click=lambda: dialog.submit(None)).props('round icon=close color=negative')
                
                benchmark = ui.input('Benchmarks to run ', placeholder='Name or ID of benchmark')
                ui.button('Select Benchmark', on_click=show_solver_selection)
            ui.button('REDIRECT', on_click=lambda e: ui.open(yet_another_page, e.socket))
            ui.card_section()


@ui.page('/yet_another_page')
def yet_another_page():
    ui.label('Welcome to yet another page')
    ui.button('RETURN', on_click=lambda e: ui.open('#open', e.socket))


        

def build_experiment_config_form() -> None:

    with ui.row().classes('flex justify-center w-full'):
        ui.add_head_html('<h3 align="center">Experiment Configuration </h3>')
        
        build_general_settings()
        ui.card()
        ui.card()
        


            
            


            
            #passwcord = ui.input('Password').classes('w-full').props()

def table_in_dialog() -> None:
   
    with ui.card().classes('flex justify-center w-1/4'):
            table = ui.table(table_options).classes('max-h-40')
            
            with ui.row().classes('flex justify-center w-full'):
                ui.button('Confirm')
                ui.button('Cancel')
















@ui.page('/')
def main_page() -> None:
    build_experiment_config_form()
    #table_in_dialog()


ui.run(title='probo2-webinterface')
           