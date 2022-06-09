import src.functions.register as register
import shutil

def zip(directory):

    shutil.make_archive(directory, 'zip',directory)

def tar(directory):

    shutil.make_archive(directory, 'tar',directory)



register.archive_functions_register('zip',zip)

register.archive_functions_register('tar',tar)