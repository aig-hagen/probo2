"""Custom click classes and callbacks"""
import os
import ast

import click

from src import definitions


class TrackToProblemClass(click.Option):
    def type_cast_value(self, ctx, value):
        value = value.strip().split(",")
        tasks = []
        for track in value:
            if track in definitions.SUPPORTED_TRACKS.keys():
             tasks.extend(definitions.SUPPORTED_TRACKS[track])
        return tasks


def command_required_option_from_option(require_name):
    class CommandOptionRequiredClass(click.Command):

        def invoke(self, ctx):
            guess = ctx.params[require_name]
            if not guess:
                if ctx.params['tasks'] is None or ctx.params['format'] is None:
                    raise click.ClickException(
                        "With {}={} must specify option --{} and --{}".format(
                            require_name, guess, "problems", "format"))
            super(CommandOptionRequiredClass, self).invoke(ctx)

    return CommandOptionRequiredClass


class StringAsOption(click.Option):
    def type_cast_value(self, ctx, value):
        if not value:
            return []
        else:
            return value.split(",")


class FilterAsDictionary(click.Option):

    def type_cast_value(self, ctx, value):
        if not value:
            return {}
        else:
            filter_dic = dict()
            for f in value:
                click.echo(f)
                f_split = f.split(":")
                filter_dic[f_split[0]] = ast.literal_eval(f_split[1])
            return filter_dic


class StringToInteger(click.Option):

    def type_cast_value(self, ctx, value):
        if value == 'None':
            return []
        if not value:
            return []
        else:
            int_string = value.split(",")
            numbers = list(map(int, int_string))
            return numbers


def call_click_command(cmd, *args, **kwargs):
    """ Wrapper to call a click command

    :param cmd: click cli command function to call 
    :param args: arguments to pass to the function 
    :param kwargs: keywrod arguments to pass to the function 
    :return: None 
    """

    # Get positional arguments from args
    arg_values = args[0]
    print("ARG_values:", arg_values)
    args_needed = {c.name: c for c in cmd.params
                   if c.name not in arg_values}

    # build and check opts list from kwargs
    opts = {a.name: a for a in cmd.params if isinstance(a, click.Option)}
    print("opts:", opts)
    for name in kwargs:
        if name in opts:
            arg_values[name] = kwargs[name]
        else:
            if name in args_needed:
                arg_values[name] = kwargs[name]
                del args_needed[name]
            else:
                raise click.BadParameter(
                    "Unknown keyword argument '{}'".format(name))

    # check positional arguments list
    for arg in (a for a in cmd.params if isinstance(a, click.Argument)):
        if arg.name not in arg_values:
            raise click.BadParameter("Missing required positional"
                                     "parameter '{}'".format(arg.name))

    # build parameter lists
    opts_list = sum(
        [[o.opts[0], str(arg_values[n])] for n, o in opts.items()], [])
    args_list = [str(v) for n, v in arg_values.items() if n not in opts]

    # call the command
    cmd(opts_list + args_list)


# callbacks
def check_path(ctx, param, value):
    """ Checks if a user input path is valid.

    This function is a callback function of the command-line-interface and is call.
    Args:
        ctx: Context of click applications
        param: Passed parameters from command line
        value: Value of the parameter
    Returns:
        True if the path is valid.
    Raises:
        click.BadParameter: Path not found

    """
    if not os.path.exists(value):
        raise click.BadParameter("Path not found!")
    else:
        return value


def check_problems(ctx, param, value):
    """ Checks if problems are supported.

     This function is a callback function of the command-line-interface
     Args:
         ctx: Context of click applications
         param: Passed parameters from command line
         value: Value of the parameter
    Returns:
        List of problems as strings.
     Raises:
         click.BadParameter: Problem is not supported
     """
    if not value:
        return []
    else:
        value = value.strip()
        problems = value.split(",")
    for problem in problems:
        if problem not in definitions.SUPPORTED_TASKS:
            raise click.BadParameter("Problem not supported!")
    return problems


def check_ids(ids):
    pass
