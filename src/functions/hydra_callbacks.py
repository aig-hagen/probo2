from hydra.core.plugins import Plugins
from hydra.core.singleton import Singleton
from hydra.core.utils import JobReturn
from typing import List

import logging

from hydra.core.utils import JobReturn
from typing import Any, List
from hydra.experimental.callback import Callback
from omegaconf import DictConfig


class AggregateResults(Callback):
    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:

        print("AggregateResults!")
        print(config['index_file'])
        # log.info("All jobs are complete. Aggregating results...")
        # results = [job.return_value for job in jobs if job.status == "COMPLETED"]
        # log.info(f"Aggregated results: {results}")
        # with open("results_summary.yaml", "w") as f:
        #     import yaml
        #     yaml.dump(results, f)


log = logging.getLogger(__name__)

class MyCallback(Singleton):
    def on_multirun_end(self, jobs: List[JobReturn]) -> None:

        print("Callback started!")
        # log.info("All jobs are complete. Aggregating results...")
        # results = [job.return_value for job in jobs if job.status == "COMPLETED"]
        # log.info(f"Aggregated results: {results}")
        # with open("results_summary.yaml", "w") as f:
        #     import yaml
        #     yaml.dump(results, f)

