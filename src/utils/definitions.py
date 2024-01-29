import src.probo2_data
import os
import src.probo2_data.json
import src.probo2_data.solver_test_data


import src.generators
import src.experiment_configs
import src.results
import src.webinterface
import src.probo2_data.markdown
from importlib_resources import files
# Defaults for paths
ROOT_DIR = os.path.abspath(os.curdir)  # Root of project
SRC_DIR = os.path.join(ROOT_DIR,"src")
DATABASE_DIR = files(src.probo2_data).joinpath(".probo2")
TEST_DATABASE_PATH = os.path.join(str(DATABASE_DIR), "probo2_old.db")
CONFIGS_DIRECTORY = str(files(src.experiment_configs))
RESULT_DIRECTORY = str(files(src.results))
EXPERIMENT_INDEX = str(files(src.probo2_data).joinpath("experiment_index.csv"))
WEB_INTERFACE_FILE = str(files(src.webinterface).joinpath("web_interface.py"))


# Solver interface test files
TEST_INSTANCES_REF_PATH = str(files(src.probo2_data.solver_test_data).joinpath("reference"))
TEST_INSTANCE_ARG = str(files(src.probo2_data.solver_test_data).joinpath("a.arg"))
TEST_INSTANCE_ARG_I23 = str(files(src.probo2_data.solver_test_data).joinpath("a.arg.i23"))
TEST_INSTANCE_APX = str(files(src.probo2_data.solver_test_data).joinpath("a.apx"))
TEST_INSTANCE_TGF = str(files(src.probo2_data.solver_test_data).joinpath("a.tgf"))
TEST_INSTANCE_I23 = str(files(src.probo2_data.solver_test_data).joinpath("a.i23"))
TEST_INSTANCE_APXM = str(files(src.probo2_data.solver_test_data).joinpath("a.apxm"))
TEST_INSTANCE_TGFM = str(files(src.probo2_data.solver_test_data).joinpath("a.tgfm"))


CSS_TEMPLATES_PATH = str(files(src.probo2_data).joinpath("css"))


# JSON files

SOLVER_FILE_PATH = str(files(src.probo2_data.json).joinpath('solvers.json'))
BENCHMARK_FILE_PATH = str(files(src.probo2_data.json).joinpath('benchmarks.json'))
STATUS_FILE_DIR = str(files(src.probo2_data.json).joinpath(".probo2_status.json"))
PLOT_JSON_DEFAULTS =  str(files(src.probo2_data.json).joinpath("plotting_defaults.json"))
FETCH_BENCHMARK_DEFAULTS_JSON = str(files(src.probo2_data.json).joinpath('probo2_fetch_benchmark_defaults.json'))
GENERATOR_DEFAULTS_JSON = str(files(src.generators).joinpath('generator_config.json'))

LOG_FILE_PATH = str(files(src.probo2_data).joinpath('probo2_log.txt'))
DEFAULT_CONFIG_PATH = str(os.path.join(CONFIGS_DIRECTORY,'default_config.yaml'))
LAST_EXPERIMENT_JSON_PATH = str(files(src.probo2_data).joinpath('probo2_last_experiment.json'))
LAST_EXPERIMENT_SUMMARY_JSON_PATH = str(files(src.probo2_data).joinpath('probo2_last_experiment_summary.json'))

# YAML files
PRETTY_LATEX_TABLE_CONFIG =  str(files(src.probo2_data).joinpath('pretty_latex_table_config.yaml'))

# Markdown files
MARK_DOWN_ROOT = files(src.probo2_data.markdown)
SOLVERS_MD = str(MARK_DOWN_ROOT.joinpath('solvers.md'))

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

from enum import Enum

class DefaultInstanceFormats(Enum):
  APX = 'apx'
  TGF = 'tgf'
  I23 = 'i23'

  def as_list():
    return [f.value for f in DefaultInstanceFormats]

class DefaultQueryFormats(Enum):
  ARG = 'arg'

  def as_list():
    return [f.value for f in DefaultQueryFormats]

class DefaultReferenceExtensions(Enum):
  REF = 'ref'
  OUT = 'out'
  RES = 'res'

  def as_list():
    return [f.value for f in DefaultReferenceExtensions]


