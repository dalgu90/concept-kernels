import logging
import os

import hydra
from omegaconf import OmegaConf, DictConfig

from concept_kernels.datasets import load_dataset
from concept_kernels.models import build_model
from concept_kernels.trainers import build_trainer
from concept_kernels.utils.config_utils import add_custom_resolvers, format_output_dir
from concept_kernels.utils.logger import init_logger
from concept_kernels.utils.misc import seed


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(config: DictConfig):
    # Format and create output directory
    format_output_dir(config)
    OmegaConf.resolve(config)
    os.makedirs(config.output_dir, exist_ok=True)

    # Create a custom logger
    logger = init_logger(config.output_dir, "train" if not config.test else "test")
    logger.info(OmegaConf.to_yaml(config))   # log the config

    if not config.test:  # Training
        # Save the config
        OmegaConf.save(config, os.path.join(config.output_dir, "config.yaml"))

        # Random seed
        seed(config.seed)

        # Load datasets, model, trainer
        train_dataset = load_dataset(conf=config.dataset, split="train", seed=config.seed)
        val_dataset = load_dataset(conf=config.dataset, split="val", seed=config.seed)
        kg, metadata = train_dataset.kg, train_dataset.metadata
        model = build_model(conf=config.model, kg=kg, metadata=metadata)
        trainer = build_trainer(conf=config.trainer)

        # Train!
        trainer.train(model, train_dataset, val_dataset)
    else:  # Testing
        # Data, model, trainer
        if config.test_all_split:
            train_dataset = load_dataset(conf=config.dataset, split="train", seed=config.seed)
            val_dataset = load_dataset(conf=config.dataset, split="val", seed=config.seed)
        test_dataset = load_dataset(conf=config.dataset, split="test", seed=config.seed)
        kg, metadata = test_dataset.kg, test_dataset.metadata
        model = build_model(conf=config.model, kg=kg, metadata=metadata)
        trainer = build_trainer(conf=config.trainer)

        # Test!
        if config.test_all_split:
            trainer.test(model, train_dataset)
            trainer.test(model, val_dataset)
        trainer.test(model, test_dataset)
    print("done")

if __name__ == '__main__':
    add_custom_resolvers()
    main()
