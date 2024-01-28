"""Handle user cli input"""

import src.functions.register as register

def get_user_input():
    print("Input graph:")
    lines = []
    while True:
        try:
            line = input()
            lines.append(line)
        except EOFError:
            break
    return lines

def get_input_format(user_input):
    """Determine the graph format of input"""
    if "#" in user_input:
        return 'tgf'
    if 'arg' in user_input[0]:
        return 'apx'
    
    return None
       