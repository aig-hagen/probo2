
from omegaconf import DictConfig
def get_iccma23_interface_call_static(config: DictConfig):
    """
    Constructs the command-line parameters for invoking the ICCMA23 solver interface.
    This function generates a list of command-line parameters based on the provided configuration.
    It determines the appropriate interpreter (bash or python) based on the solver file extension
    and appends optional parameters for the solver execution.
    Args:
        config (DictConfig): A configuration object containing the solver path, task, and optional arguments.
    Returns:
        list: A list of command-line parameters to be used for invoking the solver.
    """
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


    # Check if there are any additional paramters specified in cfg.solver.arguments
    if 'argument' in config.solver:
        # Check if the arguments are present in the experiment sweep params if so append them to the command
        for argument in config.solver.argument:
            #print(f'Argument: {argument=} {config[argument]=}')
            if argument in config:
                cmd_params.extend([argument, config[argument]])

    return cmd_params

def get_legacy_interface_static(config: DictConfig):
    """
    Constructs a list of command parameters for the legacy solver interface based on the provided configuration.

    Args:
        config (DictConfig): A configuration object containing solver settings and parameters.

    Returns:
        list: A list of command parameters to be used for the solver interface.
    """
    cmd_params = get_iccma23_interface_call_static(config)

    cmd_params.extend(['-fo',config.solver.format])

    # Check if there are any additional paramters specified in cfg.solver.arguments
    if 'argument' in config.solver:
        # Check if the arguments are present in the experiment sweep params
        for argument in config.solver.argument:
            if argument in config:
                cmd_params.extend([argument, config[argument]])

    return cmd_params

interfaces_dict = {'i23': get_iccma23_interface_call_static, 'legacy': get_legacy_interface_static}
