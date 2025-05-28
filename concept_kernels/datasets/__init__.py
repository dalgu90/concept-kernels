from concept_kernels.datasets.talent_dataset import TalentDataset


def load_dataset(conf, split, seed=0):
    if conf.name == 'talent_dataset':
        CLS = TalentDataset
    else:
        raise ValueError(f"Wrong dataset name: {conf.name}")

    assert split in conf.params, f"Wrong split: {split}"
    params = conf.params[split]

    return CLS(**params, split=split, seed=seed)
