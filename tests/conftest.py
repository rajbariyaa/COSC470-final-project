import copy

import pytest

from fedheart.data import download_raw, load_dataset
from fedheart.splits import make_split
from fedheart.utils import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def dataset(cfg):
    download_raw(cfg)  # verifies checksums and record counts; downloads only missing files
    return load_dataset(cfg)


@pytest.fixture(scope="session")
def split0(cfg, dataset):
    return make_split(dataset, 0, cfg["split"]["fractions"])


@pytest.fixture
def tmp_cfg(cfg, tmp_path):
    c = copy.deepcopy(cfg)
    c["paths"]["results_dir"] = str(tmp_path / "results")
    return c
