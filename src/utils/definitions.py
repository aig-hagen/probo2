import os
# Defaults for paths
ROOT_DIR = os.path.abspath(os.curdir)  # Root of project
SRC_DIR = os.path.join(ROOT_DIR,"src")
DATABASE_DIR = os.path.join(ROOT_DIR, 'databases')
TEST_DATABASE_PATH = os.path.join(DATABASE_DIR, "probo2_old.db")
PLOT_JSON_DEFAULTS = os.path.join(ROOT_DIR, os.path.join("src", "plotting", "defaults.json"))
TEST_INSTANCES_REF_PATH = os.path.join(ROOT_DIR, os.path.join("src", "test", "reference"))
CSS_TEMPLATES_PATH = os.path.join(SRC_DIR,"css")

STATUS_FILE_DIR = os.path.join(ROOT_DIR,"status.json")



SUPPORTED_TASKS = ["EE-ST", "EE-CO", "EE-PR", "EE-GR", "EE-SST", "EE-STG",
                      "SE-ST", "SE-SST", "SE-STG", "SE-CO", "SE-PR", "SE-GR", "SE-ID",
                      "DC-ST", "DC-SST", "DC-STG", "DC-CO", "DC-PR", "DC-ID", "DC-GR",
                      "DS-ST", "DS-SST", "DS-STG", "DS-CO", "DS-PR", "DS-GR"]

SUPPORTED_TRACKS = {'CO': ['EE-CO', 'SE-CO', 'DC-CO', 'DS-CO'],
                    'GR': ['EE-GR','SE-GR', 'DC-GR', 'DS-GR'],
                    'PR': ['EE-PR','SE-PR','DC-PR','DS-PR'],
                    'ST':  ['EE-ST', 'SE-ST', 'DC-ST', 'DS-ST']
                    }
