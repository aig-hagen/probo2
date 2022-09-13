"""Custom click classes and callbacks"""
import os
import ast

import click


from src.utils import definitions



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

def command_required_tag_if_not(require_name):
    class CommandOptionRequiredClass(click.Command):

        def invoke(self, ctx):
            last = ctx.params[require_name]
            if not last and not ctx.params['name']:
                    raise click.BadOptionUsage(option_name='name',message=f"With option --{require_name}={last} you must specify a value for option --tag.")

            super(CommandOptionRequiredClass, self).invoke(ctx)

    return CommandOptionRequiredClass

class StringAsOption(click.Option):
    def type_cast_value(self, ctx, value):
        if not value:
            return None
        else:
            splitted =  value.split(",")
            final_values = []
            for v in splitted:
                if v.isdigit():
                    final_values.append(int(v))
                else:
                    final_values.append(v)
            return final_values



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


def check_problems(ctx, param, value, in_db=True):
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
    return problems


def check_ids(ids):
    pass


import click.formatting as cl_formmating
from contextlib import contextmanager

class MyFormatter(cl_formmating.HelpFormatter):
    def __init__(self, indent_increment=2, width=None, max_width=None):
        super().__init__(indent_increment=indent_increment, width=width, max_width=max_width)

    def write_usage(self, prog, args="", prefix="*Usage: "):
        """Writes a usage line into the buffer.

        :param prog: the program name.
        :param args: whitespace separated list of arguments.
        :param prefix: the prefix for the first line.
        """
        usage_prefix = "{:>{w}}{}*".format(prefix, prog, w=self.current_indent)
        text_width = self.width - self.current_indent

        if text_width >= (cl_formmating.term_len(usage_prefix) + 20):
            # The arguments will fit to the right of the prefix.
            indent = " " * cl_formmating.term_len(usage_prefix)
            self.write(
                cl_formmating.wrap_text(
                    args,
                    text_width,
                    initial_indent=usage_prefix,
                    subsequent_indent=indent,
                )
            )
        else:
            # The prefix is too long, put the arguments on the next line.
            self.write(usage_prefix)
            self.write("\n")
            indent = " " * (max(self.current_indent, cl_formmating(prefix)) + 4)
            self.write(
                cl_formmating.wrap_text(
                    args, text_width, initial_indent=indent, subsequent_indent=indent
                )
            )

        self.write("\n")

    def write_text(self, text):
        """Writes re-indented text into the buffer.  This rewraps and
        preserves paragraphs.
        """
        text_width = max(self.width - self.current_indent, 11)
        indent = " " * self.current_indent
        self.write(
            cl_formmating.wrap_text(
                f'{text}',
                text_width,
                initial_indent=indent,
                subsequent_indent=indent,
                preserve_paragraphs=True,
            )
        )
        self.write("\n")
    def write_dl(self, rows, col_max=30, col_spacing=2):
        """Writes a definition list into the buffer.  This is how options
        and commands are usually formatted.

        :param rows: a list of two item tuples for the terms and values.
        :param col_max: the maximum width of the first column.
        :param col_spacing: the number of spaces between the first and
                            second column.
        """
        rows = list(rows)
        widths = cl_formmating.measure_table(rows)
        if len(widths) != 2:
            raise TypeError("Expected two columns for definition list")

        for first, second in cl_formmating.iter_rows(rows, len(widths)):

            self.write("{:>}{}".format("",f'+ *{first}*\n\n'))
            if not second:
                self.write("\n")
                continue
            else:
                self.write(f'    {second}\n')

    @contextmanager
    def section(self, name):
        """Helpful context manager that writes a paragraph, a heading,
        and the indents.

        :param name: the section name that is written as heading.
        """
        self.write_paragraph()
        self.write_heading(f'**{name}**')
        self.indent()
        try:
            yield
        finally:
            self.dedent()

