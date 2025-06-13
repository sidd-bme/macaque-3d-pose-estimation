"""
step1_proc2d.py

Markerless Macaque 2D-to-3D Pose Pipeline: 
Detection (Swin-MaskRCNN, MMDetection), Tracking (BoTSORT, BoxMOT), 
Pose Estimation (ViTPose, MMPose), and ID Classification (ResNet, MMPREtrain).

Author: Siddharth K Nagaraj
Lab: System Emotional Science, University of Toyama
Original: based on code from Dr. Jumpei Matsumoto
Last Update: 2025-06-05

Tested Environment:
  - Ubuntu 20.04.6
  - Python 3.9.21
  - CUDA GPU (Nvidia A5000)
  - mmengine==0.10.7, mmcv==2.1.0, mmdet==3.2.0, mmpose==1.3.2, boxmot==12.0.7, mmpretrain==1.2.0

Usage: 
    from step1_proc2d import init_all_models, process_single_cam, init_id_model
    # See README for details.
"""
# Imports & Environment Logging
import os
import json
import glob
import logging
from pathlib import Path
from collections import deque

import cv2
import torch
import imgstore
import numpy as np
from tqdm import tqdm

from mmcv.transforms import Compose
from mmengine.logging import print_log
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model as init_pose_model, inference_topdown
from mmpretrain.utils import register_all_modules as register_pretrain_modules
from mmpretrain import ImageClassificationInferencer
from boxmot.trackers.botsort.botsort import BotSort
import warnings
#surpress numba warning
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration & Hyperparameters

DEVICE = "cuda:1"

DETECT_CONFIG     = "./model/detection/SWIN-Mask_R-CNN_bbox_only.py"
DETECT_CHECKPOINT = "./model/detection/detection.pth"

POSE_CONFIG       = "./model/pose/td-hm_ViTPose-huge_8xb64-210e_coco-256x192_sn_macaque.py"
POSE_CHECKPOINT   = "./model/pose/pose.pth"

ID_CONFIGS = {
    "normal": "./model/id/sn_resnet152_8xb32_in1k_pretrained_optimized_finetuned.py",
    "mff1y":  "./model/id/sn_resnet152_8xb32_in1k_mff1y_pretrained_optimized.py",
}
ID_CKPTS = {
    "normal": "./model/id/id_finetuned.pth",
    "mff1y":  "./model/id/id_mff1y.pth",
}

SCORE_THR    = 0.85    # Detection score threshold
KP_THR       = 0.30    # Pose keypoint threshold
EMA_ALPHA    = 0.50    # EMA weight for smoothing
DISP_THR     = 20.0    # EMA displacement threshold (px/frame)
MIN_MARGIN   = 0.20    # Min box margin (%)
MAX_MARGIN   = 0.50    # Max box margin (%)
DESIRED_AR   = 192.0 / 256.0  # Target aspect ratio (0.75)
ID_CONF_THR  = 0.80    # Minimum ID confidence
TRACK_BUFFER = 72      # Frames to buffer lost tracks (~3s @24fps)

BOTSORT_CFG = dict(
    reid_weights=None,
    device=torch.device(DEVICE),
    half=False,
    with_reid=False,
    track_high_thresh=SCORE_THR,
    track_low_thresh=0.10,
    new_track_thresh=SCORE_THR,
    track_buffer=TRACK_BUFFER,
    match_thresh=0.80,
    frame_rate=24,
    cmc_method='sift'
)

smoothing_buffer: dict[int, deque] = {}

##MODEL INIT

def init_all_models(device: str = DEVICE, id_variant: str = "normal"):
    """Load detection, tracking, pose, and ID models once."""
    print_log(f"[INFO] Initializing models on device: {device}", logger="current")
    detector = init_detector(DETECT_CONFIG, DETECT_CHECKPOINT, device=device)
    tracker = BotSort(**BOTSORT_CFG)
    pose_model = init_pose_model(POSE_CONFIG, POSE_CHECKPOINT, device=device)
    pose_model.test_cfg = dict(flip_test=True, flip_mode="heatmap", shift_heatmap=False)

    ##Swin-MaskRCNN test pipeline (bbox-only)
    test_pipeline = Compose([
        {"type": "mmdet.LoadImageFromNDArray"},
        {"type": "Resize", "scale": (800, 800), "keep_ratio": True},
        {"type": "mmdet.LoadAnnotations", "with_bbox": True},
        {"type": "mmdet.PackDetInputs"},
    ])

    register_pretrain_modules(init_default_scope=True)
    id_config = ID_CONFIGS.get(id_variant)
    id_ckpt = ID_CKPTS.get(id_variant)
    id_model = None
    if id_config and id_ckpt:
        id_model = ImageClassificationInferencer(
            model=id_config,
            pretrained=id_ckpt,
            device=device,
        )
    print_log("[INFO] All models initialized.", logger="current")
    return detector, tracker, pose_model, test_pipeline, id_model


def init_id_model(device: str, id_variant: str):
    """Load the dedicated ID model for a single camera."""
    register_pretrain_modules(init_default_scope=True)
    id_config = ID_CONFIGS.get(id_variant)
    id_ckpt = ID_CKPTS.get(id_variant)
    if id_config and id_ckpt:
        return ImageClassificationInferencer(
            model=id_config,
            pretrained=id_ckpt,
            device=device,
        )
    return None

##UTILITY FUNCTIONS

def classify_patches(id_model, patches: list[np.ndarray], input_size: int = 224) -> list[dict]:
    """
    Crop-and-classify patches through the ID model.
    Returns a list of {pred_label, pred_score} for each patch.
    """
    if id_model is None or not patches:
        return [{"pred_label": -1, "pred_score": 0.0} for _ in patches]

    valid, idx_map = [], []
    for i, patch in enumerate(patches):
        h, w = patch.shape[:2]
        if h > 0 and w > 0:
            resized = cv2.resize(patch, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
            valid.append(resized)
            idx_map.append(i)

    if not valid:
        return [{"pred_label": -1, "pred_score": 0.0} for _ in patches]

    raw_results = id_model(valid, batch_size=len(valid))
    out = [{"pred_label": -1, "pred_score": 0.0} for _ in patches]
    for idx, res in zip(idx_map, raw_results):
        out[idx] = {"pred_label": int(res["pred_label"]), "pred_score": float(res["pred_score"])}
    return out

##MAIN PROCESSING LOOP
def process_single_cam(
    store,
    out_dir: str,
    T: np.ndarray,
    detector,
    tracker,
    pose_model,
    test_pipeline,
    id_model,
):
    """
    Run detection -> tracking -> pose -> ID classification -> EMA smoothing.
    Save `alldata.json` and `frame_num.npy` under `output`.
    """
    smoothing_buffer.clear()
    missed_detection_count = 0
    missed_track_count = 0
    print_log(f"[INFO] Output dir: {out_dir}", logger="current")
    os.makedirs(out_dir, exist_ok=True)
    alldata_path = Path(out_dir) / "alldata.json"
    fnums_path = Path(out_dir) / "frame_num.npy"
    if alldata_path.exists() and fnums_path.exists():
        print_log(f"[skip] {out_dir} already processed", logger="current")
        return

    first_img, _ = store.get_image(frame_number=None, frame_index=0)
    md = store.get_frame_metadata()
    t_cam, fnums = md["frame_time"], md["frame_number"]

    results_all, fnums_out = [], []
    frame_number = -1
    kp_params = {
        "score_thr": SCORE_THR,
        "kp_thr": KP_THR,
        "ema_alpha": EMA_ALPHA,
        "disp_thr": DISP_THR,
        "min_margin": MIN_MARGIN,
        "max_margin": MAX_MARGIN,
        "desired_ar": DESIRED_AR,
        "id_conf_thr": ID_CONF_THR,
    }
    #maxlen = no. of predictions for buffer to 'remember'
    #id_vote_buffers: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=15))

    print_log(f"[INFO] Starting frame processing: {len(T)} frames.", logger="current")
    for t in tqdm(T, desc=os.path.basename(store.filename)):
        idx = int(np.abs(t_cam - t).argmin())
        if frame_number >= fnums[idx]:
            results_all.append(results_all[-1] if results_all else [])
            fnums_out.append(frame_number)
            continue

        ##Advance to the next req frame
        if frame_number == -1:
            img, (frame_number, _) = store.get_image(fnums[idx])
        else:
            while frame_number < fnums[idx]:
                img, (frame_number, _) = store.get_next_image()

        ##Detection
        det = inference_detector(detector, [img], test_pipeline=test_pipeline)[0]
        boxes_all = det.pred_instances.bboxes.cpu().numpy()
        scores_all = det.pred_instances.scores.cpu().numpy()
        keep = scores_all > kp_params["score_thr"]
        if not keep.any():
            missed_detection_count += 1
            print_log(f"[WARN] No valid detections in frame {frame_number}", logger="current", level=logging.WARNING)
            results_all.append([])
            fnums_out.append(frame_number)
            continue

        boxes, scores = boxes_all[keep], scores_all[keep]

        ##Tracking
        dets6 = np.hstack([boxes, scores[:, None], np.zeros((len(scores), 1))])
        tracks = tracker.update(dets6, img)
        if len(tracks) == 0:
            # Only count as missed if not the first frame processed
            if len(fnums_out) > 0:
                missed_track_count += 1
            print_log(f"[WARN] No valid tracks in frame {frame_number}", logger="current", level=logging.WARNING)
            results_all.append([])
            fnums_out.append(frame_number)
            continue

        tracks = np.asarray(tracks)
        boxes, tids = tracks[:, :4], tracks[:, 4].astype(int)

        ##Filter degenerate boxes
        valid_boxes, valid_tids = [], []
        for (x1, y1, x2, y2), tid in zip(boxes, tids):
            xi1, yi1, xi2, yi2 = map(int, (x1, y1, x2, y2))
            if xi2 > xi1 and yi2 > yi1:
                valid_boxes.append((xi1, yi1, xi2, yi2))
                valid_tids.append(tid)
        if not valid_boxes:
            print_log(f"[WARN] No valid bounding boxes after filtering in frame {frame_number}", logger="current")
            results_all.append([])
            fnums_out.append(frame_number)
            continue

        boxes = np.array(valid_boxes, dtype=np.int32)
        tids = np.array(valid_tids, dtype=np.int32)

        ##Dynamic margin expansion & aspect-ratio correction
        expanded_xywh = []
        for (x1, y1, x2, y2) in boxes:
            w, h = float(x2 - x1), float(y2 - y1)
            cx, cy = x1 + 0.5 * w, y1 + 0.5 * h
            frac = np.clip((h - 50.0) / (200.0 - 50.0), 0.0, 1.0)
            margin_pct = kp_params["max_margin"] - (kp_params["max_margin"] - kp_params["min_margin"]) * frac
            w_new, h_new = w * (1 + margin_pct), h * (1 + margin_pct)
            current_ar = w_new / h_new
            if abs(current_ar - kp_params["desired_ar"]) > 0.20:
                if current_ar < kp_params["desired_ar"]:
                    w_new = h_new * kp_params["desired_ar"]
                else:
                    h_new = w_new / kp_params["desired_ar"]
            expanded_xywh.append([cx, cy, w_new, h_new])
        expanded_xywh = np.array(expanded_xywh, dtype=np.float32)

        ##Pose inference (top-down)
        person_results = []
        for tid, (cx, cy, w_c, h_c) in zip(tids, expanded_xywh):
            x1_c, y1_c = cx - 0.5 * w_c, cy - 0.5 * h_c
            x2_c, y2_c = cx + 0.5 * w_c, cy + 0.5 * h_c
            person_results.append({"track_id": int(tid), "bbox": [x1_c, y1_c, x2_c, y2_c]})

        pose_res_list = inference_topdown(
            pose_model, img,
            bboxes=np.array([r["bbox"] for r in person_results], dtype=np.float32),
            bbox_format="xyxy",
        )

        ##ID classification
        patches = [img[y1:y2, x1:x2] for (x1, y1, x2, y2) in boxes]
        id_preds = classify_patches(id_model, patches)

        ##Build per-frame JSON
        frame_json = []
        for i_box, pr in enumerate(pose_res_list):
            tid = int(tids[i_box])
            kpt_xy = pr.pred_instances.keypoints[0].copy()
            try:
                kpt_score = pr.pred_instances.keypoint_scores[0].copy()
            except AttributeError:
                kpt_score = np.ones(17, dtype=np.float32)

            ##Zero out low-confidence joints
            low_conf = kpt_score < kp_params["kp_thr"]
            kpt_xy[low_conf, :2] = np.nan
            kpt_score[low_conf] = 0.0

            ##EMA smoothing
            kp_array = np.concatenate([kpt_xy, kpt_score.reshape(-1, 1)], axis=1)
            buf = smoothing_buffer.setdefault(tid, deque(maxlen=5))
            buf.append((frame_number, kp_array.copy()))

            if len(buf) >= 2:
                (f_prev, kp_prev), (f_curr, kp_curr) = buf[-2], buf[-1]
                valid_prev = ~np.isnan(kp_prev[:, 0])
                valid_curr = ~np.isnan(kp_curr[:, 0])
                valid_both = valid_prev & valid_curr
                disp = np.zeros(kp_prev.shape[0], dtype=np.float32)
                if valid_both.any():
                    disp[valid_both] = np.linalg.norm(
                        kp_curr[valid_both, :2] - kp_prev[valid_both, :2], axis=1
                    )
                smooth_mask = (disp < kp_params["disp_thr"]) & valid_both
                for j in np.where(smooth_mask)[0]:
                    kp_curr[j, :2] = (
                        kp_params["ema_alpha"] * kp_prev[j, :2]
                        + (1 - kp_params["ema_alpha"]) * kp_curr[j, :2]
                    )
                buf[-1] = (f_curr, kp_curr)

            kpt_xyv_sm = smoothing_buffer[tid][-1][1]  ##(17, 3)
            keypoints_list = [[float(x), float(y), float(s)] for (x, y, s) in kpt_xyv_sm]

            ##ID assignment
            id_label = int(id_preds[i_box]["pred_label"])
            id_score = float(id_preds[i_box]["pred_score"])

            # keep ID only when the classifier is confident, otherwise ─1
            assigned_id = id_label if id_score >= kp_params["id_conf_thr"] else -1

            x1i, y1i, x2i, y2i = boxes[i_box]
            frame_json.append([
                tid,
                float(x1i), float(y1i), float(x2i), float(y2i),
                keypoints_list,
                assigned_id,
                id_score,
            ])

        results_all.append(frame_json)
        fnums_out.append(frame_number)

    ##Save valid frames only
    valid_set = set(store.get_frame_metadata()["frame_number"])
    clean_res, clean_fnums = [], []
    for res, fnum in zip(results_all, fnums_out):
        if fnum in valid_set:
            clean_res.append(res)
            clean_fnums.append(fnum)

    print_log(f"[INFO] Writing output: {alldata_path}", logger="current")
    np.save(fnums_path, np.array(clean_fnums, dtype=np.int32))
    with open(alldata_path, "w") as fp:
        json.dump(clean_res, fp)

    print_log(
        f"[SUMMARY] Camera {os.path.basename(out_dir)}: "
        f"{missed_detection_count} frames missed detections, "
        f"{missed_track_count} frames missed tracks (except first frame).",
        logger="current",
        level=logging.INFO
    )

    print_log(f"[INFO] Done: {len(clean_res)} frames processed. Output saved to {alldata_path}.", logger="current")

##MULTI-CAMERA WRAPPER

def step1_proc2d_custom(
    data_name: str,
    results_root: str,
    raw_root: str,
    fps: float = 24.0,
    t_intv=None,
    redo: bool = False,
):
    """
    Loop over all cameras for `data_name` under `raw_root`:
    - Load each imgstore
    - Determine ID variant by folder name (contains "mff1y"?)
    - Call process_single_cam(...)
    """
    pattern = os.path.join(raw_root, f"{data_name}.*", "metadata.yaml")
    meta_paths = sorted(glob.glob(pattern))
    if not meta_paths:
        raise FileNotFoundError(f'No imgstore metadata for "{data_name}" in {raw_root}')

    stores = [imgstore.new_for_filename(p) for p in meta_paths]
    md0 = stores[0].get_frame_metadata()
    t0 = md0["frame_time"][0]

    if t_intv is None:
        t_start, t_end = t0, md0["frame_time"][-1]
    else:
        t_start = t0 + t_intv[0]
        t_end = t0 + t_intv[1]

    T = np.arange(t_start, t_end, 1.0 / fps)

    ##init heavy models once
    detector, _, pose_model, test_pipeline, _ = init_all_models(DEVICE)
    torch.set_num_threads(1)

    for store in stores:
        folder_name = os.path.basename(store.filename)
        id_variant = "mff1y" if "mff1y" in folder_name.lower() else "normal"
        id_model_cam = init_id_model(DEVICE, id_variant)

        #tracker per camera to avoid ID/state bleed
        tracker = BotSort(**BOTSORT_CFG)

        cam = folder_name.split(".")[-1]
        out_dir = os.path.join(results_root, data_name + ("" if t_intv is None else f".{int(t_intv[0]):04d}-{int(t_intv[1]):04d}"), cam)
        if os.path.exists(out_dir) and not redo:
            print_log(f"[skip] {out_dir} already exists", logger="current")
            continue

        process_single_cam(
            store=store,
            out_dir=out_dir,
            T=T,
            detector=detector,
            tracker=tracker,
            pose_model=pose_model,
            test_pipeline=test_pipeline,
            id_model=id_model_cam,
        )


def proc(data_name, results_root, raw_root, device_str="cuda:1", fps=24.0):
    """Compatibility entry point for run_demo.py."""
    step1_proc2d_custom(data_name, results_root, raw_root, fps=fps)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--raw_root", default="/mnt/nas_siddharth/code_test/videos")
    parser.add_argument("--res_root", default="/mnt/nas_siddharth/code_test/results2d")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()

    interval = None
    if args.start is not None and args.end is not None:
        interval = (args.start, args.end)

    step1_proc2d_custom(
        data_name=args.data,
        results_root=args.res_root,
        raw_root=args.raw_root,
        fps=args.fps,
        t_intv=interval,
        redo=args.redo,
    )
