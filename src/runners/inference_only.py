import os
import os.path as osp
import logging
import time
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
import cv2

from dataloaders import build_dataloader
from detectors import build_detector
from trackers import build_tracker
from utils import mkdir_if_missing, draw_frame, gen_video

from .base import BaseRunner

log = logging.getLogger(__name__)

class InferenceRunner(BaseRunner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._vis_result = cfg['runner']['vis_result']
        
        # Set up dataloader for test clips only
        _, _, _, self._clip_loaders = build_dataloader(cfg)
        
    def run(self):
        detector = build_detector(self._cfg)
        tracker = build_tracker(self._cfg)
        
        for key, dataloader_and_gt in self._clip_loaders.items():
            match, clip_name = key
            dataloader = dataloader_and_gt['clip_loader']
            
            # Set up output directory for visualizations
            if self._vis_result:
                vis_frame_dir = osp.join(self._output_dir, '{}_{}'.format(match, clip_name))
                mkdir_if_missing(vis_frame_dir)
            else:
                vis_frame_dir = None
            
            log.info('Running inference on match={}, clip={}'.format(match, clip_name))
            
            # Run inference
            self._run_inference(detector, tracker, dataloader, vis_frame_dir)
    
    @torch.no_grad()
    def _run_inference(self, detector, tracker, dataloader, vis_frame_dir=None):
        tracker.refresh()
        results = {}
        
        # Process frames
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='[(CLIP-WISE INFERENCE)]')):
            # Extract images and transformation matrices
            if len(batch) == 6:  # Normal case with ground truth
                imgs, _, trans, _, _, img_paths = batch
            else:  # Modified case without ground truth
                imgs, trans = batch[:2]
                img_paths = batch[-1] if len(batch) > 2 else None
            
            # Run model forward pass
            batch_results, _ = detector.run_tensor(imgs, trans)
            
            # Process results
            img_paths = [list(in_tuple) for in_tuple in img_paths]
            for ib in batch_results.keys():
                for ie in batch_results[ib].keys():
                    img_path = img_paths[ie][ib]
                    preds = batch_results[ib][ie]
                    result = tracker.update(preds)
                    results[img_path] = result
                    
                    # Visualize if requested
                    if vis_frame_dir:
                        self._visualize_frame(img_path, result, vis_frame_dir)
        
        # Generate video if visualizations were created
        if vis_frame_dir:
            video_path = '{}.mp4'.format(vis_frame_dir)
            gen_video(video_path, vis_frame_dir, fps=25.0)
            log.info('Generated video: {}'.format(video_path))
    
    def _visualize_frame(self, img_path, result, vis_frame_dir):
        img = cv2.imread(img_path)
        x, y = result['x'], result['y']
        is_visible = result['visi']
        
        # Draw ball position
        if is_visible:
            img = cv2.circle(img, (int(x), int(y)), 8, (0, 0, 255), -1)
        
        # Save frame
        output_path = osp.join(vis_frame_dir, osp.basename(img_path))
        cv2.imwrite(output_path, img)