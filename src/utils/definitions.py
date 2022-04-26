import os
import src.data
import src.data.json
import src.data.test
import src.solver
import src.benchmarks
import src.generators
import src.configs_experiment
import src.results
from importlib_resources import files
# Defaults for paths
ROOT_DIR = os.path.abspath(os.curdir)  # Root of project
SRC_DIR = os.path.join(ROOT_DIR,"src")
DATABASE_DIR = files(src.data).joinpath(".probo2")
SOLVER_FILE_PATH = str(files(src.solver).joinpath('solvers.json'))
BENCHMARK_FILE_PATH = str(files(src.benchmarks).joinpath('benchmarks.json'))
TEST_DATABASE_PATH = os.path.join(str(DATABASE_DIR), "probo2_old.db")
CONFIGS_DIRECTORY = str(files(src.configs_experiment))
RESULT_DIRECTORY = str(files(src.results))

TEST_INSTANCES_REF_PATH = str(files(src.data.test).joinpath("reference"))
TEST_INSTANCE_ARG = str(files(src.data.test).joinpath("a.arg"))
TEST_INSTANCE_APX = str(files(src.data.test).joinpath("a.apx"))
TEST_INSTANCE_TGF = str(files(src.data.test).joinpath("a.tgf"))

CSS_TEMPLATES_PATH = str(files(src.data).joinpath("css"))

STATUS_FILE_DIR = str(files(src.data.json).joinpath(".probo2_status.json"))
PLOT_JSON_DEFAULTS =  str(files(src.data.json).joinpath("plotting_defaults.json"))
FETCH_BENCHMARK_DEFAULTS_JSON = str(files(src.data.json).joinpath('probo2_fetch_benchmark_defaults.json'))
GENERATOR_DEFAULTS_JSON = str(files(src.generators).joinpath('generator_config.json'))

LOG_FILE_PATH = str(files(src.data).joinpath('probo2_log.txt'))
DEFAULT_CONFIG_PATH = str(os.path.join(CONFIGS_DIRECTORY,'default_config.yaml'))
LAST_EXPERIMENT_JSON_PATH = str(files(src.data).joinpath('probo2_last_experiment.json'))
LAST_EXPERIMENT_SUMMARY_JSON_PATH = str(files(src.data).joinpath('probo2_last_experiment_summary.json'))

ALEMBIC_INIT_FILE_PATH = os.path.join(str(DATABASE_DIR),'alembic.ini')


SUPPORTED_TASKS = ["EE-ST", "EE-CO", "EE-PR", "EE-GR", "EE-SST", "EE-STG",
                      "SE-ST", "SE-SST", "SE-STG", "SE-CO", "SE-PR", "SE-GR", "SE-ID",
                      "DC-ST", "DC-SST", "DC-STG", "DC-CO", "DC-PR", "DC-ID", "DC-GR",
                      "DS-ST", "DS-SST", "DS-STG", "DS-CO", "DS-PR", "DS-GR","DS-ID",
                      "CE-CO", "CE-ST","CE-PR","CE-SST","CE-STG",
                        "ES-ST", "ES-SST", "ES-STG", "ES-CO", "ES-PR", "ES-GR",
                         "EC-ST", "EC-SST", "EC-STG", "EC-CO", "EC-PR", "EC-ID", "EC-GR",
                         "EE-ST-D"]

SUPPORTED_TRACKS = {'CO': ['EE-CO', 'SE-CO', 'DC-CO', 'DS-CO',"CE-CO",'EC-CO','ES-CO'],
                    'GR': ['EE-GR','SE-GR', 'DC-GR', 'DS-GR','EC-GR','ES-GR'],
                    'PR': ['EE-PR','SE-PR','DC-PR','DS-PR','CE-PR','EC-PR','ES-PR'],
                    'ST':  ['EE-ST', 'SE-ST', 'DC-ST', 'DS-ST',"CE-ST",'EC-ST','ES-ST'],
                    'SST': ["EE-SST","SE-SST","DC-SST","DS-SST","CE-SST",'EC-SST','ES-SST'],
                    'STG': ["EE-STG","SE-STG","DC-STG","DS-STG","CE-STG",'EC-STG','ES-STG'],
                    'ID':["SE-ID","DS-ID" ]
                    }
