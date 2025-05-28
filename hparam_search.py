from collections.abc import Iterable
import logging
import os
import sys

import hydra
import optuna
from omegaconf import OmegaConf, DictConfig

from concept_kernels.datasets import load_dataset
from concept_kernels.models import build_model
from concept_kernels.trainers import build_trainer
from concept_kernels.utils.config_utils import add_custom_resolvers, format_output_dir, change_output_dir
from concept_kernels.utils.logger import init_logger
from concept_kernels.utils.misc import seed


def search_update_params(trial, conf):
    def get_distribution(distribution_name):
        return getattr(trial, f'suggest_{distribution_name}')

    params = {}
    for item in conf.search.search_space:
        name, dist = item['name'], item['dist']
        if hasattr(item, 'cond'):
            if not eval('conf.' + item['cond']):
                continue

        if not isinstance(dist, Iterable):
            val = dist
        else:
            dist, *args = dist
            if dist.startswith('?'):
                val = (
                    get_distribution(dist.lstrip('?'))(name, *args[1], **args[2])
                    if get_distribution("categorical")(f"option_{name}", [True, False])
                    else args[0]
                )
            else:
                val = get_distribution(dist)(name, *args[0], **args[1])
        params[name] = val
        OmegaConf.update(conf, name, val)


@hydra.main(config_path="configs", config_name="hparam_search", version_base=None)
def main(config: DictConfig):
    # Format and create output directory
    format_output_dir(config)
    OmegaConf.resolve(config)
    os.makedirs(config.output_dir, exist_ok=True)

    # Create a custom logger
    logger = init_logger(config.output_dir, "train" if not config.test else "test")
    logger.info(OmegaConf.to_yaml(config))   # log the config

    # Create Optuna study
    optuna.logging.get_logger("optuna").propagate= True
    study_name = config.search.name
    journal_path = os.path.join(config.output_dir, f"optuna_journal_storage.log")
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(journal_path),  # NFS path for distributed optimization
    )
    if config.search.sampler == "RandomSampler":
        sampler = optuna.samplers.RandomSampler(seed=config.seed)
    elif config.search.sampler == "TPESampler":
        sampler = optuna.samplers.TPESampler(seed=config.seed)
    elif config.search.sampler == "GridSampler":
        grid = config.search.grid
        sampler = optuna.samplers.GridSampler(grid)
    else:
        raise ValueError(f"Wrong sampler: {config.search.sampler}")

    study = optuna.create_study(
        study_name=study_name,
        sampler=sampler,
        direction=config.search.direction,
        storage=storage,
        load_if_exists=True,
    )

    if not config.test:  # Search
        assert config.seed == 0, "Hparam search only uses 0 seed"
        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if n_complete >= config.search.max_trial:
            logger.info(f"Exceeded {config.search.max_trial} trials")
        else:
            def objective(trial):
                conf = config.copy()
                new_output_dir = os.path.join(conf.output_dir, f"trials/{trial.number}")
                change_output_dir(conf, new_output_dir)
                search_update_params(trial, conf)
                os.makedirs(conf.output_dir, exist_ok=True)

                # Save the config
                OmegaConf.save(conf, os.path.join(conf.output_dir, "config.yaml"))

                # Random seed
                seed(conf.seed)

                # Load datasets
                train_dataset = load_dataset(conf=conf.dataset, split="train", seed=conf.seed)
                val_dataset = load_dataset(conf=conf.dataset, split="val", seed=conf.seed)
                kg, metadata = train_dataset.kg, train_dataset.metadata

                # Load model, trainer
                model = build_model(conf=conf.model, kg=kg, metadata=metadata)
                trainer = build_trainer(conf=conf.trainer)

                # Train!
                trainer.train(model, train_dataset, val_dataset)
                trainer.test(model, train_dataset)
                trainer.test(model, val_dataset)
                metric_vals = trainer.evaluate(model, val_dataset)
                return metric_vals[conf.search.eval_metric]

            def max_trial_callback(study, trial):
                n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
                if n_complete >= config.search.max_trial:
                    study.stop()

            # Search!
            study.optimize(objective, callbacks=[max_trial_callback])
    else:  # Testing
        # Get the best model
        best_trial = study.best_trial
        conf = config.copy()
        new_output_dir = os.path.join(conf.output_dir, f"best_trial/seed{conf.seed}")
        change_output_dir(conf, new_output_dir)
        search_update_params(best_trial, conf)
        os.makedirs(conf.output_dir, exist_ok=True)
        logger.info(OmegaConf.to_yaml(conf))   # log the best config

        # Data, model, trainer
        if config.test_all_split:
            train_dataset = load_dataset(conf=conf.dataset, split="train", seed=conf.seed)
            val_dataset = load_dataset(conf=conf.dataset, split="val", seed=conf.seed)
        test_dataset = load_dataset(conf=conf.dataset, split="test", seed=conf.seed)
        kg, metadata = test_dataset.kg, test_dataset.metadata

        seed(conf.seed)

        model = build_model(conf=conf.model, kg=kg, metadata=metadata)
        trainer = build_trainer(conf=conf.trainer)

        # Train!
        trainer.train(model, train_dataset, val_dataset)

        # Test!
        if config.test_all_split:
            trainer.test(model, train_dataset)
            trainer.test(model, val_dataset)
        trainer.test(model, test_dataset)
    print("done")

if __name__ == '__main__':
    add_custom_resolvers()
    main()
