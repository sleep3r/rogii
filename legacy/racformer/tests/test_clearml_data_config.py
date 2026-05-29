from racformer.config import load_config


def test_remote_training_configs_enable_clearml_dataset_resolution() -> None:
    for path in ("configs/racformer_sanity.yml", "configs/racformer_v1.yml"):
        cfg = load_config(path)

        assert cfg.tracking.enabled is True
        assert cfg.data_clearml.enabled is True
        assert cfg.data_clearml.project == "ROGII/Wellbore"
        assert cfg.data_clearml.name == "rogii-wellbore-geology-prediction"
        assert cfg.data_clearml.version == "20260519_s3"
