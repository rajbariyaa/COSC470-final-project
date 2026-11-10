import ast
import json
from pathlib import Path

import numpy as np
import pytest

import fedheart
from fedheart.prepare import DEVELOPMENT_PARTITIONS, prepare_seed, seed_dir
from fedheart.testset import (
    ACCESS_LOG,
    SELECTION_FILE,
    SealedTestSetError,
    load_test_arrays,
    lock_selection,
)
from fedheart.utils import results_dir

DEVELOPMENT_MODULES = (
    "prepare.py", "centralized.py", "fedavg_flower.py", "fedavg_reference.py", "finetune.py",
)  # fmt: skip


def test_test_data_sealed_until_selection_locked(tmp_cfg, dataset):
    prepare_seed(tmp_cfg, dataset, 0)
    with pytest.raises(SealedTestSetError):
        load_test_arrays(tmp_cfg, dataset, 0, purpose="should fail")


def test_lock_then_access_is_logged_and_tamper_detected(tmp_cfg, dataset):
    prepare_seed(tmp_cfg, dataset, 0)
    lock_selection(tmp_cfg, {"note": "test"})
    arrays = load_test_arrays(tmp_cfg, dataset, 0, purpose="unit test")
    assert sum(len(v[2]) for v in arrays.values()) == 185
    log = (results_dir(tmp_cfg) / ACCESS_LOG).read_text().strip().splitlines()
    assert json.loads(log[-1])["purpose"] == "unit test"
    with pytest.raises(SealedTestSetError, match="already been accessed"):
        lock_selection(tmp_cfg, {"note": "changed"})
    (results_dir(tmp_cfg) / SELECTION_FILE).write_text('{"note": "tampered"}')
    with pytest.raises(SealedTestSetError, match="changed"):
        load_test_arrays(tmp_cfg, dataset, 0, purpose="tampered")


def test_client_files_contain_no_test_records(tmp_cfg, dataset):
    out = prepare_seed(tmp_cfg, dataset, 0)
    split = out["split"]
    test_ids = set(split.loc[split["partition"] == "test", "record_id"])
    for f in (seed_dir(tmp_cfg, 0) / "client_data").glob("*.npz"):
        with np.load(f) as z:
            assert not {k for k in z.files if "test" in k}
            assert not test_ids & (set(z["ids_train"]) | set(z["ids_val"]))
    assert DEVELOPMENT_PARTITIONS == ("train", "val")


def test_development_code_never_imports_test_access():
    src = Path(fedheart.__file__).parent
    for name in DEVELOPMENT_MODULES:
        tree = ast.parse((src / name).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
                imported |= {f"{node.module}.{a.name}" for a in node.names}
        assert not any("testset" in m or "load_test_arrays" in m for m in imported), name
