"""
    Checkpoint Saver
"""
import collections
import json
import os
import logging

import torch

from concept_kernels.utils.file_utils import load_json, save_json


logger = logging.getLogger(__name__)

class BaseCheckpointSaver(object):
    def __init__(self, config):
        cls_name = self.__class__.__name__
        logger.debug(f"Initializing {cls_name} with config: {config}")

        self.config = config
        self.info_fname = "ckpt-info.json"
        if hasattr(self.config, "info_fname"):
            self.info_fname = self.config.info_fname

    def save_ckpt_info(self, info):
        info_fpath = os.path.join(self.config.checkpoint_dir, self.info_fname)
        save_json(info, info_fpath)

    def load_ckpt_info(self):
        info_fpath = os.path.join(self.config.checkpoint_dir, self.info_fname)
        if os.path.exists(info_fpath):
            # Load json (convert int string to int)
            info = load_json(info_fpath)
            info["iter_ckpts"] = {
                int(i): fname for i, fname in info["iter_ckpts"].items()
            }
            return info
        else:
            return {"best_ckpt": None, "iter_ckpts": {}}

    def clean_up_ckpt_info(self, info):
        # Check best ckpt
        if info["best_ckpt"]:
            ckpt_fpath = os.path.join(
                self.config.checkpoint_dir, info["best_ckpt"]["fname"]
            )
            if not os.path.exists(ckpt_fpath):
                info["best_ckpt"] = None

        # Check iter based ckpts
        train_iters_to_delete = []
        for train_iter, ckpt_fname in info["iter_ckpts"].items():
            ckpt_fpath = os.path.join(self.config.checkpoint_dir, ckpt_fname)
            if not os.path.exists(ckpt_fpath):
                train_iters_to_delete.append(train_iter)

        for train_iter in train_iters_to_delete:
            del info["iter_ckpts"][train_iter]

        return info

    def get_latest_checkpoint(self):
        info = self.load_ckpt_info()
        info = self.clean_up_ckpt_info(info)
        if info["iter_ckpts"]:
            max_iter = max(info["iter_ckpts"].keys())
            return max_iter, info["iter_ckpts"][max_iter]
        return None

    def get_best_checkpoint(self):
        info = self.load_ckpt_info()
        info = self.clean_up_ckpt_info(info)
        return info["best_ckpt"]["fname"] if info["best_ckpt"] else None

    def check_interval(self, train_iter):
        return train_iter % self.config.interval == 0

    def check_best(self, metric_val=None):
        # Compare with the best metric value
        info = self.load_ckpt_info()
        info = self.clean_up_ckpt_info(info)
        if info["best_ckpt"]:
            desired = self.config.desired
            if self.config.metric.name not in info["best_ckpt"]:
                metric = list(info["best_ckpt"].keys())
                metric.remove("fname")
                raise ValueError(
                    f"best_ckpt has metric {metric}, not"
                    f"self.config.metric.name"
                )
            best_val = info["best_ckpt"][self.config.metric.name]
            is_best = (desired == "max" and best_val < metric_val) or (
                desired == "min" and best_val > metric_val
            )
        else:
            is_best = True

        return is_best

    def save_ckpt(
        self,
        model,
        train_iter,
        optimizer=None,
        is_best=False,
        metric_val=None,
        ckpt_fname=None,
    ):
        # Load ckpt info
        info = self.load_ckpt_info()
        info = self.clean_up_ckpt_info(info)

        # New checkpoint data and name
        state_dict = collections.OrderedDict()
        for k, v in model.state_dict(keep_vars=True).items():
            if v.requires_grad:
                state_dict[k] = v.detach()
        checkpoint = {
            "model": state_dict,
            # "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "encoders": model.dumps_encoders(),
        }
        if not ckpt_fname:
            if is_best:
                ckpt_fname = self.config.best_fname_format.format(train_iter)
            else:
                ckpt_fname = self.config.ckpt_fname_format.format(train_iter)
        ckpt_fpath = os.path.join(self.config.checkpoint_dir, ckpt_fname)

        # Save new ckpt
        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)

        if is_best:
            # For best, delete the old one and save the new one
            assert metric_val is not None
            if info["best_ckpt"]:
                old_ckpt_fpath = os.path.join(
                    self.config.checkpoint_dir, info["best_ckpt"]["fname"]
                )
                logger.debug(f"Removing ckpt {old_ckpt_fpath}")
                os.remove(old_ckpt_fpath)
            logger.debug(f"Saving ckpt to {ckpt_fpath}")
            torch.save(checkpoint, ckpt_fpath)
            info["best_ckpt"] = {
                "fname": ckpt_fname,
                self.config.metric.name: metric_val,
            }
        else:
            # For iter based ckpt, save first and delete excessive ones
            info["iter_ckpts"] = {
                i: fname
                for i, fname in info["iter_ckpts"].items()
                if fname != ckpt_fname
            }
            logger.debug(f"Saving ckpt to {ckpt_fpath}")
            torch.save(checkpoint, ckpt_fpath)
            info["iter_ckpts"][train_iter] = ckpt_fname

            train_iters_del = sorted(info["iter_ckpts"].keys(), reverse=True)
            train_iters_del = train_iters_del[self.config.max_to_keep :]
            for i in train_iters_del:
                old_ckpt_fpath = os.path.join(
                    self.config.checkpoint_dir, info["iter_ckpts"][i]
                )
                logger.debug(f"Removing ckpt {old_ckpt_fpath}")
                os.remove(old_ckpt_fpath)
                del info["iter_ckpts"][i]

        # Save ckpt info
        self.save_ckpt_info(info)
        return ckpt_fname

    def load_ckpt(self, model, ckpt_fname, optimizer=None, ignore_modules=None):
        ckpt_fpath = os.path.join(self.config.checkpoint_dir, ckpt_fname)
        logger.debug(f"Loading ckpt from {ckpt_fpath}")
        checkpoint = torch.load(ckpt_fpath, map_location='cpu', weights_only=False)
        model_state_dict = checkpoint["model"]
        if ignore_modules:
            for k in list(model_state_dict.keys()):
                if any(map(lambda pref: k.startswith(pref), ignore_modules)):
                    del model_state_dict[k]
        model.load_state_dict(model_state_dict, strict=False)
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
        model.loads_encoders(checkpoint['encoders'])

    def save_args(self, args):
        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)
        args_fpath = os.path.join(self.config.checkpoint_dir, "args.json")
        logger.debug(f"Saving arguments to {args_fpath}")
        with open(args_fpath, "w") as fd:
            json.dump(vars(args), fd)
