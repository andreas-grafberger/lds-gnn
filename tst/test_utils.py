from os.path import abspath
from pathlib import Path

import pytest
from torch_geometric.datasets import Planetoid


def resource_folder_path() -> Path:
    return Path(abspath(__file__)).parent / "res"


@pytest.fixture()
def cora():
    return Planetoid(str(resource_folder_path() / "cora"), "cora")


@pytest.fixture()
def citeseer():
    return Planetoid(str(resource_folder_path() / "citeseer"), "citeseer")
