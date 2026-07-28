"""
Copied segments from demo/demo.py and demo/predictor.py files in the
DNTextSpotter GitHub Repo:

https://github.com/yyyyyxie/DNTextSpotter
"""
import argparse
from logging import Logger
import multiprocessing as mp
import os
import json
import time
import tqdm
import torch

from detectron2.modeling import build_model
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from adet.data.augmentation import Pad
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from adet.config import get_cfg
from detectron.structures import Boxes

### Classes
class ViTAEPredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
            cfg.INPUT.MAX_SIZE_TEST
        )
        # each size must be divided by 32 with no remainder for ViTAE
        self.pad = Pad(divisible_size=32)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Parameters
        ----------
            original_image: numpy ndarray.
                An image array of shape (H, W, C) (in BGR order).

        Returns
        -------
            dict:
                the output of the model for one image only.
        """
        with torch.no_grad():# https://github.com/sphinx-doc/sphinx/issues/4258
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self\
                .aug.get_transform(original_image).apply_image(original_image)
            image = self.pad.get_transform(image).apply_image(image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions

### Functions
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    # cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    # cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    # cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    # cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="DNTextSpotter Inference")
    parser.add_argument(
        "input",
        metavar = "path/to/pngs/dir",
        action = "store",
        help = "Required. Path to the folder/directory containing the pngs."
    )
    parser.add_argument(
        "output",
        metavar = "path/to/json",
        help =\
            "Required. Directory to save model outputs, these are saved as "\
            "jsons."
    )
    parser.add_argument(
        "--config-file", metavar = "FILE", help = "path to config file"
    )
    parser.add_argument(
        "--confidence-threshold",
        type = float,
        default = 0.3,
        help = "Minimum score for instance predictions to be shown"
    )
    parser.add_argument(
        "--opts",
        default=[],
        nargs = argparse.REMAINDER,
        help = "Modify config options using the command-line 'KEY VALUE' pairs"
    )
    return parser


if __name__ == "__main__":

    mp.set_start_method("spawn", force = True)
    args = get_parser().parse_args()
    logger: Logger = setup_logger()
    logger.info("Arguments: " + str(args))

    try:
        cfg = setup_cfg(args)

        # Using the VITAEPredictor
        model = ViTAEPredictor(cfg = cfg)
        
        if os.path.isdir(args.input[0]):
            args.input = [
                (fname, os.path.join(args.input[0], fname))
                for fname in os.listdir(args.input[0])
                if fname.endswith(".png")
            ]
        else:
            raise ValueError("Directory path must be passed to input.")

        if not os.path.isdir(args.output):
            raise ValueError("Directory path must be passed to output.")

        for fname, path in tqdm.tqdm(args.input, disable = not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format = "BGR")

            # Make predictions
            start_time = time.time()
            predictions = model(img)
            logger.info("{}: detected {} instances in {:.2f}s".format(
                path, len(predictions["instances"]), time.time() - start_time
            ))
            # Unpack predictions - used to reduce memory strain
            predictions = predictions["instances"]
            predictions = predictions.to("cpu")
            predictions = predictions.get_fields()
            for k in predictions.keys():
                if isinstance(predictions[k], torch.Tensor):
                    predictions[k] = predictions[k].tolist()
                elif isinstance(predictions[k], Boxes):
                    predictions[k] = predictions[k].tensor.tolist()

            # Save output 
            out_dir = os.path.join(args.output, fname.replace(".png", ".json"))
            with open(out_dir, mode = "w") as f:
                json.dump(predictions, f)
    
    except Exception as e:
        logger.error(
            f"Error encountered at {fname}: {repr(e)}", exc_info = True
        )
        raise
