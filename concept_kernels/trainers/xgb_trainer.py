import logging
import os

import numpy as np
import xgboost as xgb

from concept_kernels.modules import (
    load_graph_writer,
    load_metric,
)
from concept_kernels.utils.file_utils import save_json, save_pickle
from concept_kernels.utils.misc import show_trainable_params_xgb

logger = logging.getLogger(__name__)


class RecordEvalCallback(xgb.callback.TrainingCallback):
    def __init__(self, eval_dataset, eval_fn, writer=None):
        super().__init__()
        self.history = []
        self.eval_dataset = eval_dataset
        self.eval_fn = eval_fn
        self.writer = writer

    def after_iteration(self, model, epoch, evals_log):
        metric_vals = self.eval_fn(model, self.eval_dataset)
        eval_split = self.eval_dataset.split
        logger.info(f"[Iter {epoch:4d}] Evaluating on {eval_split:>7} dataset")
        for metric_name, metric_val in metric_vals.items():
            metric_vals[metric_name] = metric_val
            logger.info(f"{metric_name:>20}: {metric_val:6f}")
            self.writer.write_scalar(
                f"val/{metric_name}", metric_val, step=epoch
            )
        self.history.append(metric_vals)
        return False


class XGBTrainer:
    def __init__(self, config):
        cls_name = self.__class__.__name__
        logger.info(f"Initializing {cls_name}")
        self.config = config

        self.eval_metrics = {}
        for metric_conf in self.config.eval_metrics:
            metric_name = metric_conf.name
            self.eval_metrics[metric_name] = load_metric(metric_conf)

    def train(self, model, train_dataset, val_dataset):
        # Dataset
        X_train, y_train = train_dataset.X, train_dataset.y
        X_val, y_val = val_dataset.X, val_dataset.y

        # Graph writer, callback
        writer = load_graph_writer(self.config.graph.writer)
        callback = RecordEvalCallback(
            eval_dataset=val_dataset,
            eval_fn=self.evaluate,
            writer=writer,
        )
        model.callbacks = [callback]

        # Train!
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Print model parameters
        show_trainable_params_xgb(model)

        # Save the model
        ckpt_fname = "xgb.ubj"
        ckpt_fpath = os.path.join(self.config.output_dir, ckpt_fname)
        model.save_model(ckpt_fpath)
        logger.info(f"Checkpoint saved to {ckpt_fname}")

    def test(self, model, test_dataset):
        """Load the best or latest ckpt and evalutate on the given dataset."""
        # If the dataset is empty, return.
        if test_dataset is None or len(test_dataset) == 0:
            logger.info("The dataset is empty. Exit.")
            return

        # Load the model
        ckpt_fname = "xgb.ubj"
        ckpt_fpath = os.path.join(self.config.output_dir, ckpt_fname)
        ckpt_exists = True
        if hasattr(model, 'checkpoint_exists'):
            if not model.checkpoint_exists(ckpt_fpath):
                ckpt_exists = False
        elif not os.path.exists(ckpt_fpath):
            ckpt_exists = False

        if not ckpt_exists:
            logger.error("Cannot find the model checkpoint")
            return

        model.load_model(ckpt_fpath)
        logger.info(f"Loaded checkpoint from {ckpt_fname}")

        # Print model parameters
        show_trainable_params_xgb(model)

        test_split = test_dataset.split
        logger.info(f"Evaluating on {test_split:>7} dataset")
        metric_vals, epoch_outputs = self.evaluate(model, test_dataset, return_output=True)

        # Print and save results
        for metric_name, metric_val in metric_vals.items():
            logger.info(f"{metric_name:>20}: {metric_val:6f}")

        result_fname = f"{test_split}_result.json"
        result_fpath = os.path.join(self.config.output_dir, result_fname)
        logger.info(f"Saving result on {result_fpath}")
        save_json(metric_vals, result_fpath)

        if self.config.test_save_output:
            output_fname = f"{test_split}_output.pkl"
            output_fpath = os.path.join(self.config.output_dir, output_fname)
            logger.info(f"Saving model output to {output_fpath}")
            save_pickle(epoch_outputs, output_fpath)

    def evaluate(self, model, dataset, return_output=False):
        """Evaluate the model on the given dataset."""
        if isinstance(model, xgb.core.Booster):
            outputs = model.predict(dataset.to_dmatrix())
        else:
            outputs = model.predict(dataset.X)
        labels = dataset.y

        # Evaluate the predictions using self.eval_metrics
        metric_vals = self._compute_metrics(outputs, labels)
        if return_output:
            return metric_vals, (outputs, labels)
        else:
            return metric_vals

    def _compute_metrics(self, outputs, labels, metric_names=None):
        """
        Compute the metrics of given names.
        """
        metric_vals = {}
        if metric_names is None:
            metric_names = self.eval_metrics.keys()

        outputs_np = np.array(outputs)
        labels_np = np.array(labels)

        for metric_name in metric_names:
            metric_vals[metric_name] = self.eval_metrics[metric_name](
                y_true=labels_np, y_pred=outputs_np
            )
        return metric_vals
