# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
from functools import partial
import argparse
import logging
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Set, Literal, TypeVar, Union, Tuple, NamedTuple
from pathlib import Path
from json import load as load_json
import torch
import torch.nn as nn
import itertools
from torch.nn.parallel import DistributedDataParallel

from detectron2.checkpoint.c2_model_loading import align_and_update_state_dicts

import detectron2.utils.comm as comm
from detectron2.data import (
    MetadataCatalog,
    DatasetCatalog,
    build_detection_train_loader,
    build_detection_test_loader
)
from detectron2.engine import (
    DefaultTrainer, default_setup, hooks, launch
)
from detectron2.utils.events import EventStorage
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.modeling import GeneralizedRCNNWithTTA, build_model
from detectron2.utils.logger import setup_logger

from adet.data.dataset_mapper import DatasetMapperWithBasis
from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer
from adet.evaluation import TextEvaluator
from adet.modeling import swin, vitae_v2

import loralib as lora

M = TypeVar("M")

TORCH_VERSION: Tuple[int, ...] = tuple(int(x) for x in torch.__version__.split(".")[:2])
if TORCH_VERSION >= (1, 11):
    from torch.ao import quantization
    from torch.ao.quantization import FakeQuantizeBase, ObserverBase
elif (
    TORCH_VERSION >= (1, 8)
    and hasattr(torch.quantization, "FakeQuantizeBase")
    and hasattr(torch.quantization, "ObserverBase")
):
    from torch import quantization
    from torch.quantization import FakeQuantizeBase, ObserverBase

TORCH_VERSION: Tuple[int, ...] = tuple(int(x) for x in torch.__version__.split(".")[:2])

class _IncompatibleKeys(
    NamedTuple(
        "IncompatibleKeys",
        [
            ("missing_keys", List[str]),
            ("unexpected_keys", List[str]),
            ("incorrect_shapes", List[Tuple[str, Tuple[int], Tuple[int]]]),
        ],
    )
):
    pass


def _strip_prefix_if_present(state_dict: Dict[str, Any], prefix: str) -> None:
    """
    Strip the prefix in metadata, if any.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
        return

    for key in keys:
        newkey = key[len(prefix) :]
        state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata, if any..
    try:
        metadata = state_dict._metadata  # pyre-ignore
    except AttributeError:
        pass
    else:
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix) :]
            metadata[newkey] = metadata.pop(key)


def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--datasets-dir",
        action = "store",
        type = str,
        metavar = "path/to/datasets",
        default = ".",
        dest = "dataset_root",
        help = "Path to training datasets location."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def get_oshistoric(
    path_to_data: Union[str, Path], split: Literal["train", "tsst"]
) -> List[dict]:
    """
    Function loads specified `split` for the custom OS Historic dataset
    """
    # set-up image root path
    img_root = Path(path_to_data).joinpath(f"{split}_images").absolute()
    if not img_root.exists():
        raise ValueError(
            f"Path to OS Historic {split} split images does not exist: "\
            f"{str(img_root)}"
        )
    # load json
    with open(Path(path_to_data).joinpath(f"{split}.json"), "r") as f:
        data: list[dict] = load_json(f)
    # adjust paths to images
    for img_dict in data:
        img_dict["file_name"] = str(img_root.joinpath(img_dict["file_name"]))

    return data

def replace_layers(model: M) -> M:
    """
    Replaces attention layers and output components MLP Layers with
    LoRA integrated layers.
    """
    for name_, module in model.named_children():
        if name_ == "attn":
            module.qkv = lora.MergedLinear(
                *module.qkv.weight.T.shape,
                r = 8,
                enable_lora = [True, False, True]
            )
            module.proj = lora.Linear(*module.proj.weight.T.shape, r = 8)

        elif name_ in ("self_attn", "attn_cross"):
            module.attention_weights = lora.MergedLinear(
                *module.attention_weights.weight.T.shape,
                r = 8,
                enable_lora = [True]
            )
            module.value_proj = lora.Linear(
                *module.value_proj.weight.T.shape, r = 8
            )

        elif name_ in ("bezier_coord_embed", "bezier_proposal_coord"):
            for i in range(2):
                module.layers[i] = lora.Linear(
                    *module.layers[i].weight.T.shape, r = 8
                )

        elif name_ in ("ctrl_point_coord", "boundary_offset"):
            for i in range(6):
                for j in range(2):
                    module[i].layers[j] = lora.Linear(
                        *module[i].layers[j].weight.T.shape, r = 8
                    )
        
        elif sum(1 for _ in module.children()) > 0:
            ## apply recursively## simple module
            try:
                model[int(name_)] = replace_layers(module)
            except:
                setattr(model, name_, replace_layers(module))

    return model


class LoRACheckPointer(AdetCheckpointer):
    """
    Checkpointer for LoRA trained models.
    """
    def __init__(self, model, base_w_pth, save_dir="", *, save_to_disk=None, **checkpointables):
        super().__init__(
            model,
            save_dir,
            save_to_disk = save_to_disk,
            **checkpointables,
        )
        self._base_path = base_w_pth


    def resume_or_load(self, path: str, *, resume: bool = True) -> Dict[str, Any]:
        """
        If `resume` is True, this method attempts to resume from the last
        checkpoint, if exists. Otherwise, load checkpoint from the given path.
        This is useful when restarting an interrupted training job.

        Args:
            path (str): path to the checkpoint.
            resume (bool): if True, resume from the last checkpoint if it exists
                and load the model together with all the checkpointables. Otherwise
                only load the model without loading any checkpointables.

        Returns:
            same as :meth:`load`.
        """
        if resume and self.has_checkpoint():
            path = self.get_checkpoint_file()
            self.load(self._base_path, checkpointables = [])
            return self.load(path)
        elif not os.path.samefile(path, self._base_path):
            # Assume new path is a path to LoRA weights only
            self.load(self._base_path, checkpointables = [])
            return self.load(path, checkpointables = [])
        else:
            return self.load(path, checkpointables = [])


    def _load_model(self, checkpoint: Any) -> _IncompatibleKeys:
        """
        Load weights from a checkpoint.

        Args:
            checkpoint (Any): checkpoint contains the weights.

        Returns:
            ``NamedTuple`` with ``missing_keys``, ``unexpected_keys``,
                and ``incorrect_shapes`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys
                * **incorrect_shapes** is a list of (key, shape in checkpoint, shape in model)

            This is just like the return value of
            :func:`torch.nn.Module.load_state_dict`, but with extra support
            for ``incorrect_shapes``.
        """
        if checkpoint.get("matching_heuristics", False):
            self._convert_ndarray_to_tensor(checkpoint["model"])
            # convert weights by name-matching heuristics
            checkpoint["model"] = align_and_update_state_dicts(
                self.model.state_dict(),
                checkpoint["model"],
                c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
            )

        # for non-caffe2 models, use standard ways to load it
        checkpoint_state_dict = checkpoint.pop("model")
        self._convert_ndarray_to_tensor(checkpoint_state_dict)

        # if the state_dict comes from a model that was wrapped in a
        # DataParallel or DistributedDataParallel during serialization,
        # remove the "module" prefix before performing the matching.
        _strip_prefix_if_present(checkpoint_state_dict, "module.")

        # workaround https://github.com/pytorch/pytorch/issues/24139
        model_state_dict = self.model.state_dict()
        incorrect_shapes = []
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                model_param = model_state_dict[k]
                # Allow mismatch for uninitialized parameters
                if TORCH_VERSION >= (1, 8) and isinstance(
                    model_param, nn.parameter.UninitializedParameter
                ):
                    continue
                shape_model = tuple(model_param.shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    has_observer_base_classes = (
                        TORCH_VERSION >= (1, 8)
                        and hasattr(quantization, "ObserverBase")
                        and hasattr(quantization, "FakeQuantizeBase")
                    )
                    if has_observer_base_classes:
                        # Handle the special case of quantization per channel observers,
                        # where buffer shape mismatches are expected.
                        def _get_module_for_key(
                            model: torch.nn.Module, key: str
                        ) -> torch.nn.Module:
                            # foo.bar.param_or_buffer_name -> [foo, bar]
                            key_parts = key.split(".")[:-1]
                            cur_module = model
                            for key_part in key_parts:
                                cur_module = getattr(cur_module, key_part)
                            return cur_module

                        cls_to_skip = (
                            ObserverBase,
                            FakeQuantizeBase,
                            lora.Linear,
                            lora.MergedLinear
                        )
                        target_module = _get_module_for_key(self.model, k)
                        if isinstance(target_module, cls_to_skip):
                            # Do not remove modules with expected shape mismatches
                            # them from the state_dict loading. They have special logic
                            # in _load_from_state_dict to handle the mismatches.
                            continue

                    incorrect_shapes.append((k, shape_checkpoint, shape_model))
                    checkpoint_state_dict.pop(k)
        incompatible = self.model.load_state_dict(checkpoint_state_dict, strict=False)
        incompatible = _IncompatibleKeys(
            missing_keys=incompatible.missing_keys,
            unexpected_keys=incompatible.unexpected_keys,
            incorrect_shapes=incorrect_shapes,
        )

        model_buffers = dict(self.model.named_buffers(recurse=False))
        for k in ["pixel_mean", "pixel_std"]:
            # Ignore missing key message about pixel_mean/std.
            # Though they may be missing in old checkpoints, they will be correctly
            # initialized from config anyway.
            if k in model_buffers:
                try:
                    incompatible.missing_keys.remove(k)
                except ValueError:
                    pass
        for k in incompatible.unexpected_keys[:]:
            # Ignore unexpected keys about cell anchors. They exist in old checkpoints
            # but now they are non-persistent buffers and will not be in new checkpoints.
            if "anchor_generator.cell_anchors" in k:
                incompatible.unexpected_keys.remove(k)
        return incompatible


    def save(self, name: str, **kwargs: Any) -> None:
        """
        Dump model and checkpointables to a file.

        Args:
            name (str): name of the file.
            kwargs (dict): extra arbitrary data to save.
        """
        if not self.save_dir or not self.save_to_disk:
            return

        data = {}
        data["model"] = lora.lora_state_dict(self.model)
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = "{}.pth".format(name)
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        self.logger.info("Saving checkpoint to {}".format(save_file))
        with self.path_manager.open(save_file, "wb") as f:
            torch.save(data, f)
        self.tag_last_checkpoint(basename)


class LoRATrainer(DefaultTrainer):
    """
    This is the same Trainer as the DefaultTrainer except that we
    rewrite the `build_train_loader`/`resume_or_load` method.
    """
    def build_hooks(self):
        """
        Replace `DetectionCheckpointer` with `AdetCheckpointer`.

        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        """
        ret = super().build_hooks()
        for i in range(len(ret)):
            if isinstance(ret[i], hooks.PeriodicCheckpointer):
                self.checkpointer = LoRACheckPointer(
                    self.model,
                    self.cfg.MODEL.WEIGHTS,
                    self.cfg.OUTPUT_DIR,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )
                ret[i] = hooks.PeriodicCheckpointer(
                    self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD
                )
        return ret
    
    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger("adet.trainer")
        # param = sum(p.numel() for p in self.model.parameters())
        # logger.info(f"Model Params: {param}")
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
            self.after_train()

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        lora.mark_only_lora_as_trainable(self.model)
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        model = replace_layers(model = model)
        model.to(torch.device(cfg.MODEL.DEVICE))
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        mapper = DatasetMapperWithBasis(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        mapper = DatasetMapperWithBasis(cfg, False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if evaluator_type == "text":
            return TextEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("adet.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    @classmethod
    def build_optimizer(cls, cfg, model):
        def match_name_keywords(n, name_keywords):
            out = False
            for b in name_keywords:
                if b in n:
                    out = True
                    break
            return out

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY

            if match_name_keywords(key, cfg.SOLVER.LR_BACKBONE_NAMES):
                lr = cfg.SOLVER.LR_BACKBONE
            elif match_name_keywords(key, cfg.SOLVER.LR_LINEAR_PROJ_NAMES):
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.LR_LINEAR_PROJ_MULT

            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]


        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet")

    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = LoRATrainer.build_model(cfg)
        LoRACheckPointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = LoRATrainer.test(cfg, model) # d2 defaults.py
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(LoRATrainer.test_with_TTA(cfg, model))
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = LoRATrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":

    arg_parser = default_argument_parser()
    args = arg_parser.parse_args()
    print("Command Line Args:", args)

    # Register custom datasets
    get_oshistoric_train = partial(get_oshistoric, args.dataset_root, "train")
    get_oshistoric_test = partial(get_oshistoric, args.dataset_root, "test")
    DatasetCatalog.register("oshistoric_train", get_oshistoric_train)
    DatasetCatalog.register("oshistoric_test", get_oshistoric_test)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
