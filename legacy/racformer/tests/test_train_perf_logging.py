import torch
from racformer.train import (
    _cuda_memory_mb,
    _dataloader_kwargs,
    _device_label,
    _format_perf_suffix,
)


def test_dataloader_kwargs_enables_persistent_workers_only_with_workers() -> None:
    assert _dataloader_kwargs(batch_size=16, num_workers=0, device=torch.device("cpu")) == {
        "batch_size": 16,
        "num_workers": 0,
        "pin_memory": False,
    }

    assert _dataloader_kwargs(batch_size=16, num_workers=4, device=torch.device("cuda")) == {
        "batch_size": 16,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
    }


def test_format_perf_suffix_includes_breakdown_batches_and_memory() -> None:
    suffix = _format_perf_suffix(
        train_dt=10.25,
        val_dt=2.5,
        ema_dt=0.75,
        train_batches=3,
        val_batches=1,
        cuda_mem_mb=128.3,
    )

    assert "train_dt=10.2s" in suffix
    assert "val_dt=2.5s" in suffix
    assert "ema_dt=0.8s" in suffix
    assert "batches=3/1" in suffix
    assert "cuda_mem=128MB" in suffix


def test_device_label_and_memory_are_stable_on_cpu() -> None:
    assert _device_label(torch.device("cpu")) == "cpu"
    assert _cuda_memory_mb(torch.device("cpu")) == 0.0
