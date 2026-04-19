#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import hydra
from omegaconf import DictConfig

from het_control.run import get_experiment


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="reverse_transport_iddpg_config",
)
def hydra_experiment(cfg: DictConfig) -> None:
    experiment = get_experiment(cfg=cfg)
    experiment.run()


if __name__ == "__main__":
    hydra_experiment()
