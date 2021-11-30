import os
import src.data
import src.data.json
import src.data.test
from importlib_resources import files
# Defaults for paths
ROOT_DIR = os.path.abspath(os.curdir)  # Root of project
SRC_DIR = os.path.join(ROOT_DIR,"src")
DATABASE_DIR = files(src.data).joinpath("databases")
TEST_DATABASE_PATH = os.path.join(DATABASE_DIR, "probo2_old.db")

TEST_INSTANCES_REF_PATH = files(src.data.test).joinpath("reference")
CSS_TEMPLATES_PATH = files(src.data).joinpath("css")

STATUS_FILE_DIR = files(src.data.json).joinpath("status.json")
PLOT_JSON_DEFAULTS =  files(src.data.json).joinpath("plotting_defaults.json")

LOG_FILE_PATH = files(src.data).joinpath('log.txt')



SUPPORTED_TASKS = ["EE-ST", "EE-CO", "EE-PR", "EE-GR", "EE-SST", "EE-STG",
                      "SE-ST", "SE-SST", "SE-STG", "SE-CO", "SE-PR", "SE-GR", "SE-ID",
                      "DC-ST", "DC-SST", "DC-STG", "DC-CO", "DC-PR", "DC-ID", "DC-GR",
                      "DS-ST", "DS-SST", "DS-STG", "DS-CO", "DS-PR", "DS-GR",
                      "CE-CO", "CE-ST","CE-PR","CE-SST","CE-STG",
                        "ES-ST", "ES-SST", "ES-STG", "ES-CO", "ES-PR", "ES-GR",
                         "EC-ST", "EC-SST", "EC-STG", "EC-CO", "EC-PR", "EC-ID", "EC-GR"]

SUPPORTED_TRACKS = {'CO': ['EE-CO', 'SE-CO', 'DC-CO', 'DS-CO',"CE-CO",'EC-CO','ES-CO'],
                    'GR': ['EE-GR','SE-GR', 'DC-GR', 'DS-GR','EC-GR','ES-GR'],
                    'PR': ['EE-PR','SE-PR','DC-PR','DS-PR','CE-PR','EC-PR','ES-PR'],
                    'ST':  ['EE-ST', 'SE-ST', 'DC-ST', 'DS-ST',"CE-ST",'EC-ST','ES-ST'],
                    'SST': ["EE-SST","SE-SST","DC-SST","DS-SST","CE-SST",'EC-SST','ES-SST'],
                    'STG': ["EE-STG","SE-STG","DC-STG","DS-STG","CE-STG",'EC-STG','ES-STG']
                    }
