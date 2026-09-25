"""
Performs inference on ToponymExtractor. Saves outputs as a FIFO queue.

https://github.com/SesamePaste233/ToponymExtractor/tree/main
"""

from WordSpotter.ModelWrapper import DeepSoloWrapper
from Grouper.GrouperCaller_v1 import *
from StyleEncoder.DeepFont import DeepFontEncoder, EncodeFontBatch, load_model

from WordSpotting import pyramid_scan
from Flattening import aggregate_closest_results, normalize_adhesive
from StyleEmbedding import generate_style_embeddings
from ToponymsAssignment import group_toponyms, toponym_from_graph_strong_component

import os
from PIL import Image
import time
from pandas import DataFrame
from json import loads as load_json_string

from Utils import result_reader as rr
from Utils.bezier_utils import make_result
from Utils.visualizer import PolygonVisualizer

def get_default_config():
    return {
        # INPUT
        'img_path': None, # Overwrite this

        'task_name': None, # Overwrite this if needed

        'output_dir': 'Results/', # Overwrite this if needed

        # SETTINGS (default values work well for most cases)
        'pyramid_scan_num_layers': 1, # Significantly slows down detection speed
        'pyramid_min_patch_resolution': 750, # Lower this value for maps with smaller text
        'pyramid_max_patch_resolution': 2048, # Model only run on min_patch_resolution if pyramid_scan_num_layers = 1

        'word_spotting_score_threshold': 0.6, # 0 to 1, lower this value if some words are missed
        'word_spotting_image_batch_size': 4, # For 8G VRAM. Lower this value if CUDA OOM error occurs, increase it if you have a powerful GPU

        # Save intermediate results
        'save_stacked_detection': True,
        'save_flattened_detection': True,
        'save_grouper_graph': True,
        'save_toponym_detection': True,

        'save_visualization_images': False,

        # Model paths
        'deepsolo_config_path': 'Models/config_96voc.yaml',
        'deepsolo_model_path': 'Models/finetune_v2/model.pth',
        
        'grouper_model_path': 'Models/grouper_model_v1_epoch2.pth',

        # Optional
        'generate_style_embeddings': False,
        'use_style_embeddings_in_grouping': False,
        'deepfont_encoder_path': 'Models/DeepFontEncoder_full.pth',
    }

def merge_cfgs(default_cfg, user_cfg):
    for key, value in user_cfg.items():
        if key in default_cfg:
            default_cfg[key] = value
    
    if default_cfg['use_style_embeddings_in_grouping'] and not default_cfg['generate_style_embeddings']:
        default_cfg['use_style_embeddings_in_grouping'] = False
        print('use_style_embeddings_in_grouping is set to False because generate_style_embeddings is False')

    return default_cfg

class ToponymExtractor:
    def __init__(self, config:dict):
        if 'img_path' not in config or config['img_path'] is None:
            raise ValueError('img_path not found in config')

        default_config = get_default_config()
        config = merge_cfgs(default_config, config)

        self.config = config
        self.img_path = config['img_path']
        self.task_name = config.get('task_name', default_config['task_name'])
        self.config['task_name'] = self.task_name
        self.model_cfg = config.get('deepsolo_config_path', default_config['deepsolo_config_path'])
        self.model_weights = config.get('deepsolo_model_path', default_config['deepsolo_model_path'])
        self.grouper_model_path = config.get('grouper_model_path', default_config['grouper_model_path'])
        
        self.generate_style_embeddings = config.get('generate_style_embeddings', default_config['generate_style_embeddings'])
        self.use_style_embeddings_in_grouper = config.get('use_style_embeddings_in_grouping', default_config['use_style_embeddings_in_grouping'])
        self.deepfont_encoder_path = config.get('deepfont_encoder_path', default_config['deepfont_encoder_path'])

        self.pyramid_scan_num_layers = config.get('pyramid_scan_num_layers', default_config['pyramid_scan_num_layers'])
        self.pyramid_min_patch_resolution = config.get('pyramid_min_patch_resolution', default_config['pyramid_min_patch_resolution'])
        self.pyramid_max_patch_resolution = config.get('pyramid_max_patch_resolution', default_config['pyramid_max_patch_resolution'])

        self.word_spotting_score_threshold = config.get('word_spotting_score_threshold', default_config['word_spotting_score_threshold'])
        self.word_spotting_image_batch_size = config.get('word_spotting_image_batch_size', default_config['word_spotting_image_batch_size'])

        self._validate_paths()

        if self.task_name is None:
            self.task_name = os.path.splitext(os.path.basename(self.img_path))[0]

        # Outputs
        self.output_dir = config.get('output_dir', default_config['output_dir'])
        self.output_dir = os.path.join(self.output_dir, self.task_name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.stacked_detection_path = os.path.join(self.output_dir, f'stacked_detections.json')
        self.save_stack_detection = config.get('save_stacked_detection', default_config['save_stacked_detection'])

        self.flattened_detection_path = os.path.join(self.output_dir, f'flattened_detections.json')
        self.save_flattened_detection = config.get('save_flattened_detection', default_config['save_flattened_detection'])

        self.grouper_graph_path = os.path.join(self.output_dir, f'grouper_graph.gexf')
        self.save_grouper_graph = config.get('save_grouper_graph', default_config['save_grouper_graph'])

        self.toponym_detection_path = os.path.join(self.output_dir, f'toponym_detections.json')
        self.save_toponym_detection = config.get('save_toponym_detection', default_config['save_toponym_detection'])

        self.save_visualization_images = config.get('save_visualization_images', default_config['save_visualization_images'])

        self._print_cfg()

    def _print_cfg(self):
        print('Configuration:')
        for key, value in self.config.items():
            print(f'{key}: \t\t\t{value}')

    def _validate_paths(self):
        if not os.path.exists(self.img_path):
            raise ValueError(f'img_path {self.img_path} does not exist')

        if not os.path.exists(self.model_cfg):
            raise ValueError(f'deepsolo_config_path {self.model_cfg} does not exist')

        if not os.path.exists(self.model_weights):
            raise ValueError(f'deepsolo_model_path {self.model_weights} does not exist')

        if not os.path.exists(self.grouper_model_path):
            raise ValueError(f'grouper_model_path {self.grouper_model_path} does not exist')

        if self.generate_style_embeddings and not os.path.exists(self.deepfont_encoder_path):
            raise ValueError(f'deepfont_encoder_path {self.deepfont_encoder_path} does not exist')


    def word_spotting(self):
        self.spotter = DeepSoloWrapper(self.model_cfg, self.model_weights, score_threshold=self.word_spotting_score_threshold)
        word_spotting_results = self.spotter.inference_single(Image.open(self.img_path))
        word_spotting_results = load_json_string(DataFrame(word_spotting_results).to_json())

        bezier_cs, heights, texts, scores =\
            word_spotting_results["center_bezier_pts"],\
            word_spotting_results["avg_height"],\
            word_spotting_results["text"],\
            word_spotting_results["score"]
        
        self.word_spotting_results = [
            make_result(v, heights[k], texts[k], scores[k])
            for k, v
            in bezier_cs.items()
        ]

        print(f'Word spotting done. Found {len(self.word_spotting_results)} raw word detections.')

        return None


    def run(self, print_time=True):
        print(f'\nRunning pipeline for task: \'{self.task_name}\'.')

        start_time = time.time()

        self.word_spotting()
        step1_time = time.time()
        if print_time:
            print(f'Word spotting time: {step1_time - start_time:.2f} seconds')

        return self.word_spotting_results


if __name__ == "__main__":

    
    import argparse
    import json
    import pickle
    from pathlib import Path
    
    # Parser
    parser = argparse.ArgumentParser(description = "DNTextSpotter Inference")
    parser.add_argument(
        "config",
        metavar = "path/to/config/json",
        help = "Required path to config json file."
    )
    parser.add_argument(
        "input",
        metavar = "path/to/pngs/dir",
        action = "store",
        help = "Required. Path to the folder/directory containing the pngs."
    )
    parser.add_argument(
        "output",
        metavar = "path/to/pickle",
        help =\
            "Required. Directory to save model outputs, these are saved "\
            "into a pickle file."
    )
    args = parser.parse_args()

    # Get inference configs
    cfg = Path(args.config)
    if not cfg.is_file():
        raise ValueError(
            f"config argument not recognised as a file. "\
            f"Argument: {args.config}"
        )
    elif cfg.suffix != ".json":
        raise ValueError(
            f"config argument not recognised as a JSON file, determined by "\
            f"suffix. Argument: {args.config}"
        )
    elif not cfg.exists():
        raise ValueError(
            f"config file doesn't exist, check filepath is correct. "\
            f"Argument: {args.config}"
        )
    else:
        with open(cfg, "r") as f:
            cfg = json.load(f)

    # Get image directory
    img_dir = Path(args.input)
    if not img_dir.is_dir():
        raise ValueError(
            f"input argument is not a directory. Argument: {args.input}"
        )
    elif not img_dir.exists():
        raise ValueError(
            f"Image directory doesn't exist, check path is correct. "\
            f"Argument: {args.input}"
        )
    
    # Get output directory
    out_dir = Path(args.output)
    if out_dir.suffix != ".pkl":
        raise ValueError(
            f"output argument not recognised as a pickle file, determined "\
            f"by suffix. Argument: {args.output}"
        )

    img_paths = [*img_dir.glob("*.png")]
    modes = ["wb"] + (["ab"] * (len(img_paths) - 1))
    for path, mode in zip(img_paths, modes):
        try:
            extractor = ToponymExtractor({**cfg, **{"img_path": str(path)}})
            words = {"image": path.name, "words": extractor.run()}
        except Exception as e:
            print(f"ERROR encountered for {path.name}: {repr(e)}")
            words = {"image": path.name, "error": repr(e)}
        # Save output
        with open(out_dir, mode = mode) as f:
            pickle.dump(words, f)
