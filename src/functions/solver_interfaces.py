from omegaconf import DictConfig
def get_iccma23_interface_call_static(config: DictConfig):
    cmd_params = list()

    solver_path = config.solver.path
    task = config.task
    if solver_path.endswith('.sh'):
        cmd_params.append('bash')
    elif solver_path.endswith('.py'):
        cmd_params.append('python')


    cmd_params.extend([solver_path,
          "-p", task,
          "-f", ''])

    return cmd_params

def get_legacy_interface_static(config: DictConfig):
    cmd_params = get_iccma23_interface_call_static(config)

    cmd_params.extend(['-fo',config.solver.format]) 

    return cmd_params

interfaces_dict = {'i23': get_iccma23_interface_call_static, 'legacy': get_legacy_interface_static}
