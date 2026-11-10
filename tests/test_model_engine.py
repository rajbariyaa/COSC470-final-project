import math

import numpy as np
import torch

from fedheart.engine import local_train, make_loader, train_epoch
from fedheart.model import HeartMLP, build_model, count_parameters, initial_state


def test_architecture():
    m = HeartMLP(41, 16)
    assert m.hidden.in_features == 41 and m.hidden.out_features == 16
    assert isinstance(m.act, torch.nn.ReLU)
    assert m.out.in_features == 16 and m.out.out_features == 1
    assert count_parameters(m) == 41 * 16 + 16 + 16 + 1
    assert m(torch.zeros(5, 41)).shape == (5,)


def test_initial_state_is_seeded_and_matched():
    a, b, c = initial_state(41, 16, 0), initial_state(41, 16, 0), initial_state(41, 16, 1)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not torch.equal(a["hidden.weight"], c["hidden.weight"])


def test_initial_state_leaves_global_rng_untouched():
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    initial_state(41, 16, 7)
    assert torch.equal(torch.rand(3), expected)


def _toy(n=37, d=41, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)).astype(np.float32), rng.integers(0, 2, size=n)


def test_local_training_is_deterministic():
    X, y = _toy()
    init = initial_state(41, 16, 0)
    a = local_train(init, X, y, lr=1e-3, batch_size=16, shuffle_seeds=[1, 2], hidden=16)
    b = local_train(init, X, y, lr=1e-3, batch_size=16, shuffle_seeds=[1, 2], hidden=16)
    assert all(torch.equal(a[0][k], b[0][k]) for k in a[0])
    assert a[2] == 2 * math.ceil(37 / 16)


def test_local_training_does_not_mutate_input_state():
    X, y = _toy()
    init = initial_state(41, 16, 0)
    before = {k: v.clone() for k, v in init.items()}
    local_train(init, X, y, lr=1e-2, batch_size=16, shuffle_seeds=[1], hidden=16)
    assert all(torch.equal(before[k], init[k]) for k in init)


def test_epoch_step_count():
    X, y = _toy(n=33)
    model = build_model(initial_state(41, 16, 0), 41, 16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    _, steps = train_epoch(model, make_loader(X, y, 16, 0), opt)
    assert steps == 3
