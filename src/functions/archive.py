import src.functions.register as register
import shutil
import os

def zip(directory):
    os.makedirs(directory, exist_ok=True)
    shutil.make_archive(directory, 'zip',directory)

def tar(directory):
    os.makedirs(directory, exist_ok=True)
    shutil.make_archive(directory, 'tar',directory)



register.archive_functions_register('zip',zip)

register.archive_functions_register('tar',tar)