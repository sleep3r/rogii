from __future__ import annotations

import torch
from racformer.train import EMA
from torch import nn


def test_ema_first_update_copies_current_model_weights() -> None:
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)

    ema = EMA(model, decay=0.999)
    with torch.no_grad():
        model.weight.fill_(10.0)

    ema.update(model)

    assert ema.num_updates == 1
    assert torch.equal(ema.shadow["weight"], torch.tensor([[10.0]]))


def test_ema_second_update_uses_decay() -> None:
    model = nn.Linear(1, 1, bias=False)
    ema = EMA(model, decay=0.5)

    with torch.no_grad():
        model.weight.fill_(10.0)
    ema.update(model)

    with torch.no_grad():
        model.weight.fill_(14.0)
    ema.update(model)

    assert ema.num_updates == 2
    assert torch.equal(ema.shadow["weight"], torch.tensor([[12.0]]))
