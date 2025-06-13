##step2_crossviewmatching.py
##Find and match 3D pose correspondences across cameras at keyframes.

import os
import pickle
import yaml
import json
import math
import h5py
import cv2
import numpy as np
from src.utils import multicam_toolbox as mct
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from typing import List, Optional, Tuple

##CONSTANTS + CONFIG

THR_KP = 0.1            ##Keypoint confidence threshold
ALPHA_ID = 0.2          ##Weight for ID continuity vs. geometry
CID_THR = 0.8           ##ID confidence threshold for 2D to 3D matching
P_THR_2DT = 0.8         ##Probability threshold for 2D-track labeling
MODEL_CFG = {
    "joint_num": 17,
    "spectral": True,
    "alpha_SVT": 0.5,
    "lambda_SVT": 50,
    "dual_stochastic_SVT": False,
}

##CAMERA PARAMS LOADING

def get_camparam(config_path: str) -> dict:
    """
    Load camera intrinsic/extrinsic parameters from YAML and HDF5 files.
    Returns a dict containing K, xi, D, rvecs, tvecs, and pmat for each camera.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cam_ids = cfg["camera_id"]

    intrin_path = os.path.join(os.path.dirname(config_path), "cam_intrinsic.h5")
    extrin_path = os.path.join(os.path.dirname(config_path), "cam_extrinsic_optim.h5")

    K_list, xi_list, D_list, rvecs_list, tvecs_list = [], [], [], [], []
    for cam_id in cam_ids:
        with h5py.File(intrin_path, "r") as f_intrin:
            K_list.append(f_intrin[f"/{cam_id}/K"][()])
            xi_list.append(f_intrin[f"/{cam_id}/xi"][()])
            D_list.append(f_intrin[f"/{cam_id}/D"][()])

        with h5py.File(extrin_path, "r") as f_extrin:
            rvecs_list.append(f_extrin[f"/{cam_id}/rvec"][()])
            tvecs_list.append(f_extrin[f"/{cam_id}/tvec"][()])

    ##Build projection matrices (R|t)
    pmat_list = []
    for idx, cam_id in enumerate(cam_ids):
        with h5py.File(extrin_path, "r") as f_extrin:
            rvec = f_extrin[f"/{cam_id}/rvec"][()]
            tvec = f_extrin[f"/{cam_id}/tvec"][()]
            R_mat, _ = cv2.Rodrigues(rvec)
            pmat_list.append(np.hstack([R_mat, tvec.reshape(3, 1)]))

    return {
        "camera_id": cam_ids,
        "K": K_list,
        "xi": xi_list,
        "D": D_list,
        "rvecs": rvecs_list,
        "tvecs": tvecs_list,
        "pmat": pmat_list,
    }

##PROJECTIVE ALGORITHM (PAM) HELPERS

def proj2pav(y: np.ndarray) -> np.ndarray:
    """
    Project vector y onto the probability simplex.
    Implementation based on Sorting + cumulative sum.
    """
    y = y.copy()
    y[y < 0] = 0
    if y.sum() < 1:
        return y

    u = np.sort(y)[::-1]
    sv = np.cumsum(u)
    idx = np.arange(1, len(u) + 1)
    rho = np.nonzero(u > (sv - 1) / idx)[0][-1]
    theta = max(0, (sv[rho] - 1) / (rho + 1))
    return np.maximum(y - theta, 0)


def projR(X: np.ndarray) -> np.ndarray:
    """Row-wise simplex projection."""
    for i in range(X.shape[0]):
        X[i, :] = proj2pav(X[i, :])
    return X


def projC(X: np.ndarray) -> np.ndarray:
    """Column-wise simplex projection."""
    for j in range(X.shape[1]):
        X[:, j] = proj2pav(X[:, j])
    return X


def myproj2dpam(Y: np.ndarray, tol: float = 1e-4) -> np.ndarray:
    """
    Alternating projection for 2D doubly-stochastic matrix.
    Iterates row/column projections until convergence or max 10 iterations.
    """
    X = Y.copy()
    I2 = np.zeros_like(X)
    for _ in range(10):
        X1 = projR(X + I2)
        I1 = X1 - (X + I2)
        X2 = projC(X + I1)
        I2 = X2 - (X + I1)
        if np.abs(X2 - X).sum() / X.size < tol:
            break
        X = X2
    return X

##SPECTRAL VIRTUAL TUCK MATCHING (SVT)

def matchSVT(
    S: np.ndarray,
    dimGroup: np.ndarray,
    *,
    alpha: float = 0.1,
    pselect: int = 1,
    tol: float = 5e-4,
    maxIter: int = 500,
    verbose: bool = False,
    eigenvalues: bool = False,
    _lambda: float = 50,
    mu: float = 64,
    dual_stochastic_SVT: bool = True,
) -> np.ndarray:
    """
    Solve the assignment-matching problem via SVT (Spectral Virtual Tuck).
    Returns a binary assignment matrix (match_mat) after thresholding.
    """
    N = S.shape[0]
    ##Zero diagonal and symmetrize
    S[np.arange(N), np.arange(N)] = 0
    S = (S + S.T) / 2

    X = S.copy()
    Y = np.zeros_like(S)
    W = alpha - S
    info = {}

    for iter_idx in range(maxIter):
        X0 = X.copy()
        ##Singular Value Thresholding on (Y/mu + X)
        U, s, Vh = np.linalg.svd((Y / mu) + X, full_matrices=False)
        V = Vh.conj().T
        s_thresh = np.maximum(s - (_lambda / mu), 0)
        Q = U @ np.diag(s_thresh) @ V.T

        ##Update X
        X = Q - (W + Y) / mu
        ##Zero out blocks defined by dimGroup
        for i in range(len(dimGroup) - 1):
            i0, i1 = int(dimGroup[i]), int(dimGroup[i + 1])
            X[i0:i1, i0:i1] = 0

        ##Enforce diagonal entries if pselect == 1
        if pselect == 1:
            X[np.arange(N), np.arange(N)] = 1

        ##Clamp to [0,1]
        X = np.clip(X, 0, 1)

        ##Dual stochastic projection if requested
        if dual_stochastic_SVT:
            for i in range(len(dimGroup) - 1):
                row_beg, row_end = int(dimGroup[i]), int(dimGroup[i + 1])
                for j in range(len(dimGroup) - 1):
                    col_beg, col_end = int(dimGroup[j]), int(dimGroup[j + 1])
                    if row_end > row_beg and col_end > col_beg:
                        block = X[row_beg:row_end, col_beg:col_end]
                        X[row_beg:row_end, col_beg:col_end] = myproj2dpam(block, tol=1e-2)

        X = (X + X.T) / 2  ##Symmetrize
        Y = Y + mu * (X - Q)

        pRes = np.linalg.norm(X - Q) / N
        dRes = mu * np.linalg.norm(X - X0) / N

        if verbose:
            print(f"Iter {iter_idx}: pRes={pRes:.4e}, dRes={dRes:.4e}, mu={mu:.1f}")

        if pRes < tol and dRes < tol:
            break

        if pRes > 10 * dRes:
            mu *= 2
        elif dRes > 10 * pRes:
            mu /= 2

    X = (X + X.T) / 2
    info["time"] = None  ##time tracking omitted here
    info["iter"] = iter_idx

    if eigenvalues:
        info["eigenvalues"] = np.linalg.eig(X)

    ##Binarize by threshold 0.5
    match_mat = (X > 0.5).astype(np.uint8)
    return match_mat

##DRAWING & UTILITIES

def ellipse_line(img: np.ndarray, x1: list[float], x2: list[float], mrksize: float, clr: tuple[int, int, int]):
    """
    Draw an ellipse (cylinder) between two keypoints x1, x2 to approximate a limb segment.
    """
    dx, dy = x2[0] - x1[0], x2[1] - x1[1]
    ang = 90 if dx == 0 else math.degrees(math.atan(dy / dx))
    cen = (int((x1[0] + x2[0]) / 2), int((x1[1] + x2[1]) / 2))
    length = math.hypot(dx, dy)
    cv2.ellipse(img, (cen, (int(length), int(mrksize)), ang), clr, thickness=-1)


def clean_kp(kp: List[Optional[List[float]]], ignore_score: bool = False, show_as_possible: bool = True):
    """
    Validate or discard 2D keypoints based on confidence and position bounds.
    kp: list of [x, y, score] or None. Mutates kp in-place to None or [x, y].
    """
    valid_count = sum(1 for k in kp if k is not None and k[2] > 0.3)
    for idx, point in enumerate(kp):
        if point is None:
            continue

        x, y, s = point
        #to skip eyes
        #if idx in (1, 2):
        #    kp[idx] = None
        #    continue

        if show_as_possible:
            if valid_count == 0 or np.isnan(x) or not (-1000 < x < 3000) or not (-1000 < y < 3000):
                kp[idx] = None
            else:
                kp[idx] = [x, y]
        else:
            if (not ignore_score and (s < 0.3 or np.isnan(s))) or np.isnan(x) or not (-1000 < x < 3000) or not (-1000 < y < 3000):
                kp[idx] = None
            else:
                kp[idx] = [x, y]


def draw_kps(img: np.ndarray, kp: List[Optional[List[float]]], mrksize: int, clr: Optional[Tuple[int, int, int]] = None):
    """
    Draw keypoints (circles) and limb segments (ellipses) on the image.
    """
    cm = plt.get_cmap("hsv", 36)
    kp_colors = [cm(i) for i in range(len(kp))]
    kp_con = [
    (0,2),
    (0,1),
    (2,4),
    (1,3),
    #(0,4),
    #(0,3),
    (6,8),
    (5,7),
    (8,10),
    (7,9),
    (12,14),
    (11,13),
    (14,16),
    (13,15),
    (0,17),
    (17,6),
    (17,5),
    (17,12),
    (17,11)
    ]
    
    for idx in reversed(range(len(kp))):
        if kp[idx] is not None:
            x_int, y_int = int(kp[idx][0]), int(kp[idx][1])
            color = (int(kp_colors[idx][0] * 255), int(kp_colors[idx][1] * 255), int(kp_colors[idx][2] * 255))
            c = color if clr is None else (int(clr[0] * 255), int(clr[1] * 255), int(clr[2] * 255))
            #draw circle eyes
            if idx in (1, 2):
                cv2.circle(img, (x_int, y_int), mrksize+1, c, thickness=-1)
            else:
                cv2.circle(img, (x_int, y_int), mrksize, c, thickness=-1)

    for (i1, i2) in reversed(kp_con):
        if kp[i1] is not None and kp[i2] is not None:
            c = (int(kp_colors[i1][0] * 255), int(kp_colors[i1][1] * 255), int(kp_colors[i1][2] * 255))
            c = c if clr is None else (int(clr[0] * 255), int(clr[1] * 255), int(clr[2] * 255))
            ellipse_line(img, kp[i1], kp[i2], mrksize, c)

##UNDISTORT & DEPROJECT

def undistort_points(config_path: str, i_cam: int, pos_2d: np.ndarray) -> np.ndarray:
    """
    Undistort 2D points using CV2 omnidirectional model.
    pos_2d: shape (N_kp, 2). Returns undistorted points (N_kp, 2).
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cam_id = cfg["camera_id"][i_cam]

    intrin_path = os.path.join(os.path.dirname(config_path), "cam_intrinsic.h5")
    with h5py.File(intrin_path, "r") as f_intrin:
        K = f_intrin[f"/{cam_id}/K"][()]
        xi = f_intrin[f"/{cam_id}/xi"][()]
        D = f_intrin[f"/{cam_id}/D"][()]

    pts = cv2.omnidir.undistortPoints(
        pos_2d.reshape(-1, 1, 2).astype(np.float64),
        K, D, xi, np.eye(3),
    )
    return pts.reshape(-1, 2)

def deproject(
    config_path: str,
    i_cam: int,
    P2d: np.ndarray,
    depth: float,
    camparam: Optional[dict] = None,
) -> np.ndarray:

    """
    Convert undistorted 2D points (shape: N_kp × 2) to 3D ray points at given depth.
    Returns array (N_kp, 3) in global coordinates.
    """
    if camparam is None:
        camparam = get_camparam(config_path)

    ##treat P2d as already undistorted
    p2d = P2d if P2d.ndim == 2 else P2d[np.newaxis, :]
    N_pts = p2d.shape[0]
    pts3d = np.hstack([p2d, np.ones((N_pts, 1), float)]) * depth

    R_cam = camparam["pmat"][i_cam][:, :3]
    t_cam = camparam["tvecs"][i_cam].ravel()
    R_inv = np.linalg.inv(R_cam)

    global_pts = []
    for p in pts3d:
        diff = p - t_cam
        global_pts.append(R_inv @ diff)
    return np.vstack(global_pts)

##DISTANCE BETWEEN TWO LINES

def calc_dist_btw_lines(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute minimum distance between two 3D lines (each given as 2 points + direction).
    v1, v2 arrays of shape (N_kp, 6): [x0,y0,z0, x1,y1,z1] per keypoint.
    Returns a float distance.
    """
    p1, p2 = v1[:3], v2[:3]
    d1 = (v1[3:6] - p1) / np.linalg.norm(v1[3:6] - p1)
    d2 = (v2[3:6] - p2) / np.linalg.norm(v2[3:6] - p2)
    c = np.cross(d1, d2)
    return abs(np.dot(p2 - p1, c)) / np.linalg.norm(c)

##GEOMETRY AFFINITY MATRIX

def geometry_affinity2(
    points_set: np.ndarray,
    dimGroup: np.ndarray,
    config_path: str,
    camparam: Optional[dict] = None,
) -> np.ndarray:

    """
    Compute pairwise affinity between detections across cameras based on 3D line distances.
    points_set: (M, N_kp, 3) array of 2D keypoints + scores.
    dimGroup: array of cumulative counts per camera, shape (M+1,).
    Returns an M x M affinity matrix (float).
    """
    if camparam is None:
        camparam = get_camparam(config_path)

    M, n_kp, _ = points_set.shape
    Dth2 = 150
    dist_mat = np.full((M, M), Dth2 * 2, dtype=np.float64)
    np.fill_diagonal(dist_mat, 0)

    ##Map each detection index to camera index
    cam_for_det = []
    for i in range(M):
        cam_for_det.append(np.searchsorted(dimGroup, i, side="right") - 1)

    ##Build 3D rays for each detection
    V = []
    for i_det in range(M):
        cam_idx = cam_for_det[i_det]
        undist_pts = points_set[i_det, :, :2]
        ray_near = deproject(config_path, cam_idx, undist_pts, 0.0, camparam)
        ray_far = deproject(config_path, cam_idx, undist_pts, 1000.0, camparam)
        V.append(np.hstack([ray_near, ray_far]))

    ##Keypoint confidences
    S = [points_set[i_det, :, 2] for i_det in range(M)]

    for i in range(M):
        for j in range(i + 1, M):
            if cam_for_det[i] == cam_for_det[j]:
                continue
            ##Compute per-keypoint line distances
            dists = []
            for k_idx in range(n_kp):
                if S[i][k_idx] > THR_KP and S[j][k_idx] > THR_KP:
                    d_i = calc_dist_btw_lines(V[i][k_idx], V[j][k_idx])
                    dists.append(d_i)
            if len(dists) >= 3:
                mean_dist = np.mean(dists)
                dist_mat[i, j] = mean_dist
                dist_mat[j, i] = mean_dist

    valid_entries = dist_mat < Dth2 * 2
    dm_mean = dist_mat[valid_entries].mean()
    dm_std = dist_mat[valid_entries].std()
    affinity = - (dist_mat - dm_mean) / dm_std
    affinity = 1 / (1 + np.exp(-5 * affinity))
    affinity[dist_mat > Dth2] = 0
    return affinity

##TRIANGULATION

def calc_3dpose(
    kp_2d: np.ndarray,
    config_path: str,
    camparam: Optional[dict] = None,
) -> np.ndarray:

    """
    Triangulate 3D keypoints from multi-view 2D observations.
    kp_2d: shape (n_cam, n_kp, 3) [x, y, score].
    Returns (n_kp, 3) array of 3D coordinates.
    """
    if camparam is None:
        camparam = get_camparam(config_path)

    n_cam, n_kp, _ = kp_2d.shape
    pos2d = [kp_2d[i, :, :2] for i in range(n_cam)]
    pos2d_undist = mct.undistortPoints(config_path, pos2d, omnidir=True, camparam=camparam)

    frame_use = np.ones((n_kp, n_cam), dtype=bool)
    for k_idx in range(n_kp):
        for cam_idx in range(n_cam):
            if np.isnan(pos2d[cam_idx][k_idx, 0]) or kp_2d[cam_idx, k_idx, 2] < THR_KP:
                frame_use[k_idx, cam_idx] = False

    kp3d = mct.triangulatePoints(config_path, pos2d_undist, frame_use, True, camparam=camparam)
    return kp3d

##REPROJECTION

def reproject(
    i_cam: int,
    p3d: np.ndarray,
    camparam: Optional[dict] = None,
    config_path: str = "",
) -> np.ndarray:

    """
    Project 3D points back into camera i_cam image plane.
    p3d: shape (N_pts, 3). Returns 2D points (N_pts, 2).
    """
    if camparam is None:
        camparam = get_camparam(config_path)

    K = camparam["K"][i_cam]
    xi = camparam["xi"][i_cam]
    D = camparam["D"][i_cam]
    rvec = camparam["rvecs"][i_cam]
    tvec = camparam["tvecs"][i_cam]

    pts_2d, _ = cv2.omnidir.projectPoints(
        p3d.reshape(-1, 1, 3).astype(np.float64),
        rvec, tvec, K, xi[0][0], D,
    )
    return pts_2d.reshape(-1, 2)

##MULTI-ESTIMATOR CLASS

class MultiEstimator:
    """
    Handle matching and 3D reconstruction across cameras for a single keyframe.
    """

    def __init__(self, cfg: str, debug: bool = False):
        self.cfg = cfg
        self.debug = debug

    def predict_data(
        self,
        info_dict: dict,
        show: bool = False,
        plt_id: int = 0,
        camparam: Optional[dict] = None,
        bcomb_prev: Optional[list] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:

        """
        Given a dict of detection info per camera for one frame,
        compute matched detections across cameras and their 3D poses.
        Returns (matched_list, P3d_list, bcomb_list).
        """
        if camparam is None:
            camparam = get_camparam(self.cfg)
        if bcomb_prev is None:
            bcomb_prev = []

        n_cam = len(info_dict)
        dimGroup = [0]
        cnt = 0
        for cam_id in range(n_cam):
            cnt += len(info_dict[cam_id][0])
            dimGroup.append(cnt)
        dimGroup = np.array(dimGroup)

        ##Aggregate all detections into a single list
        info_list = []
        for cam_id in range(n_cam):
            info_list.extend(info_dict[cam_id][0])

        if not info_list:
            return [], [], []

        ##Build 2D pose arrays and detection metadata
        M = len(info_list)
        n_kp = MODEL_CFG["joint_num"]
        pose2d = np.array([det["pose2d"] for det in info_list]).reshape(M, n_kp, 2)
        pose_score = np.array([det["pose2d_raw"] for det in info_list]).reshape(M, n_kp, 3)[..., 2]
        kp_mat = np.concatenate([pose2d, pose_score[..., np.newaxis]], axis=2)

        ##Map sub-detection index → camera index
        sub2cam = np.zeros(M, dtype=int)
        for idx in range(len(dimGroup) - 1):
            sub2cam[dimGroup[idx] : dimGroup[idx + 1]] = idx

        det_boxes = [det["bbox"] for det in info_list]
        det_boxids = [det["bbox_id"] for det in info_list]
        cid_list = [det["cid"] for det in info_list]

        ##Compute geometry affinity
        geo_aff = geometry_affinity2(kp_mat.copy(), dimGroup, self.cfg, camparam=camparam)

        ##Build ID continuity matrix
        cid_mat = np.zeros_like(geo_aff, dtype=np.float64)
        for i in range(M):
            for j in range(M):
                if sub2cam[i] != sub2cam[j] and cid_list[i] >= 0 and cid_list[i] == cid_list[j]:
                    cid_mat[i, j] = 1.0

        ##Build continuity from previous matches
        cont_mat = np.zeros_like(geo_aff, dtype=np.float64)
        for bc in bcomb_prev:
            for cam_idx in range(n_cam):
                for sub_idx, bid in enumerate(det_boxids):
                    if bid[0] == cam_idx and bc[cam_idx] == bid[1]:
                        cont_mat[sub_idx, sub_idx] = 1.0

        ##Combined affinity
        alpha = ALPHA_ID
        W = alpha * cid_mat + (1 - alpha) * geo_aff
        W *= (geo_aff > 0)
        W = np.nan_to_num(W)

        ##Spectral initialization
        num_person = min(4, M)
        X0 = np.random.rand(W.shape[0], num_person)
        if MODEL_CFG["spectral"]:
            eig_vals, eig_vecs = np.linalg.eig(W)
            idx_sorted = np.argsort(eig_vals)[::-1]
            if W.shape[1] >= num_person:
                X0 = eig_vecs[:, idx_sorted[:num_person]]
            else:
                X0[:, : W.shape[1]] = eig_vecs.T

        ##SVT matching
        match_mat = matchSVT(
            W,
            dimGroup,
            alpha=MODEL_CFG["alpha_SVT"],
            _lambda=MODEL_CFG["lambda_SVT"],
            dual_stochastic_SVT=MODEL_CFG["dual_stochastic_SVT"],
        )

        ##Extract matched clusters
        col_sums = match_mat.sum(axis=0)
        matched_cols = np.nonzero(col_sums > 1.9)[0]
        bin_match = match_mat[:, matched_cols] > 0.9

        matched_list = [[] for _ in range(bin_match.shape[1])]
        for sub_idx, row in enumerate(bin_match):
            if row.sum() != 0:
                pid = row.argmax()
                matched_list[pid].append(sub_idx)
        matched_list = [np.array(lst) for lst in matched_list]

        ##Helper: pick best combination of one detection per camera
        def get_best_comb(person_idxs: np.ndarray) -> np.ndarray:
            person_idxs = np.asarray(person_idxs, dtype=int)
            cam_ids = sub2cam[person_idxs]
            cam_groups = [
                person_idxs[np.where(cam_ids == cam_idx)].tolist() or [None]
                for cam_idx in range(n_cam)
            ]
            combos = list(itertools.product(*cam_groups))
            if len(combos) == 1:
                return person_idxs

            errors = []
            for combo in combos:
                kp2d = np.zeros((n_cam, n_kp, 3))
                for cam_idx, sub_idx in enumerate(combo):
                    if sub_idx is not None:
                        kp2d[cam_idx, :, :] = info_list[sub_idx]["pose2d_raw"]
                p3d = calc_3dpose(kp2d, self.cfg, camparam=camparam)
                Derrs = []
                for cam_idx, sub_idx in enumerate(combo):
                    if sub_idx is None:
                        continue
                    reproj = reproject(cam_idx, p3d, camparam=camparam, config_path=self.cfg)
                    pts2d = info_list[sub_idx]["pose2d_raw"][:, :2]
                    valid_pts = pts2d[info_list[sub_idx]["pose2d_raw"][:, 2] > THR_KP]
                    diffs = valid_pts - reproj[info_list[sub_idx]["pose2d_raw"][:, 2] > THR_KP]
                    Derrs.append(diffs)
                if Derrs:
                    all_d = np.vstack(Derrs)
                    rmse = np.sqrt((all_d**2).mean())
                else:
                    rmse = np.inf
                errors.append(rmse)

            best_idx = int(np.argmin(errors))
            best_combo = combos[best_idx]
            return np.array([i for i in best_combo if i is not None], dtype=int)

        ##Refine matches by best combinations
        refined = []
        for person in matched_list:
            best = get_best_comb(person)
            refined.append(best)
            leftover = set(person.tolist()) - set(best.tolist())
            if len(leftover) > 1:
                extra_best = get_best_comb(np.array(list(leftover), dtype=int))
                refined.append(extra_best)

        matched_list = refined

        ##If showing results, draw bounding boxes and keypoints
        if show:
            colors = [
                (0, 0, 255), (0, 255, 0), (255, 0, 0),
                (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 0, 0),
            ]
            for cam_idx in range(n_cam):
                img = info_dict[cam_idx]["image_data"]
                for pid, person in enumerate(matched_list):
                    for sub_idx in person:
                        if sub2cam[sub_idx] != cam_idx:
                            continue
                        bbox = np.array(info_list[sub_idx]["bbox"], dtype=int)
                        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colors[pid % len(colors)], 5)
                        cv2.putText(
                            img,
                            str(sub_idx),
                            (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            3.0,
                            (255, 255, 255),
                            thickness=5,
                            lineType=cv2.LINE_4,
                        )
                        kp_list = info_list[sub_idx]["pose2d_raw"].tolist()
                        clean_kp(kp_list)
                        draw_kps(img, kp_list, mrksize=3, clr=(0, 0, 0))
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.show()

            plt.imshow(W)
            plt.colorbar()
            plt.show()

        ##Compute final 3D poses and building bcomb arrays
        P3d_list, matched_list2, bcomb_list = [], [], []
        for person_idxs in matched_list:
            if person_idxs.shape[0] < 2:
                continue
            kp2d = np.zeros((n_cam, n_kp, 3))
            for sub_idx in person_idxs:
                cam_idx = sub2cam[sub_idx]
                kp2d[cam_idx, :, :] = info_list[sub_idx]["pose2d_raw"]
            pose3d = calc_3dpose(kp2d, self.cfg, camparam=camparam)
            P3d_list.append(pose3d)

            bcomb = -np.ones(n_cam, dtype=int)
            for sub_idx in person_idxs:
                cam_idx = sub2cam[sub_idx]
                bcomb[cam_idx] = info_list[sub_idx]["bbox_id"][1]
            matched_list2.append(person_idxs)
            bcomb_list.append(bcomb)

        return matched_list2, P3d_list, bcomb_list

##ID ASSIGNMENT VIA 2D TRACKLETS

def set_id_for_each_frame_of_2dtracklets(
    Cid: dict[int, np.ndarray], n_frame: int, wsize: int
) -> dict[int, np.ndarray]:
    """
    Given raw 2D-classifier outputs Cid[bbox_id] = array(n_frame),
    fill missing IDs based on window majority. Returns updated dict.
    """
    Cid2 = {k: v.copy() for k, v in Cid.items()}

    ##Intervals of each tracklet
    intervals = {}
    for k, arr in Cid.items():
        valid_idxs = np.argwhere(arr >= -1)
        intervals[k] = [valid_idxs.min(), valid_idxs.max()]

    for k, arr in Cid.items():
        ##Build one-hot matrix of shape (n_frame, 4)
        valid_ids = [0, 2, 3, 5]
        onehot = np.zeros((n_frame, len(valid_ids)), int)
        for f_idx in range(n_frame):
            if arr[f_idx] in valid_ids:
                col_idx = valid_ids.index(arr[f_idx])
                onehot[f_idx, col_idx] = 1

        labels = np.full(n_frame, -1, dtype=int)
        intervals_k = intervals[k]
        start_f, end_f = intervals_k

        ##Step 1: identify frames with high confidence
        for f_idx in range(max(start_f, wsize // 2), min(end_f, n_frame - wsize // 2)):
            window = onehot[f_idx - wsize // 2 : f_idx + wsize // 2, :]
            cnts = window.sum(axis=0)
            if cnts.sum() > 0:
                p = cnts.max() / cnts.sum()
                if p > P_THR_2DT and cnts.max() >= 12:
                    labels[f_idx] = np.argmax(cnts)

        ##Step 2: fill entire tracklet
        unique_ids = np.unique(labels[start_f : end_f + 1])
        unique_ids = unique_ids[unique_ids >= 0]

        if unique_ids.size == 0:
            ##Single global ID if confident enough
            cnt_glob = onehot.sum(axis=0)
            if cnt_glob.sum() > 0:
                pmax = cnt_glob.max() / cnt_glob.sum()
                if pmax > P_THR_2DT and cnt_glob.max() >= 12:
                    labels[:] = np.argmax(cnt_glob)
        elif unique_ids.size == 1:
            labels[:] = unique_ids[0]
        else:
            ##Multiple IDs: split by midpoint logic
            prev_id = -1
            prev_frame = 0
            for f_idx in range(n_frame):
                curr_id = labels[f_idx]
                if curr_id >= 0 and curr_id != prev_id:
                    if prev_id == -1:
                        labels[:f_idx] = curr_id
                    else:
                        ##Determine midpoint between last occurrence of prev_id and first occurrence of curr_id
                        chk_begin, chk_end = max(1, prev_frame - wsize // 2), f_idx
                        idxs_prev = np.argwhere(onehot[:, prev_id] > 0).flatten()
                        idxs_prev = idxs_prev[np.logical_and(idxs_prev >= chk_begin, idxs_prev <= chk_end)]
                        i_prev = idxs_prev.max() if idxs_prev.size > 0 else prev_frame

                        chk_begin2, chk_end2 = prev_frame, min(f_idx + wsize // 2, n_frame)
                        idxs_curr = np.argwhere(onehot[:, curr_id] > 0).flatten()
                        idxs_curr = idxs_curr[np.logical_and(idxs_curr >= chk_begin2, idxs_curr <= chk_end2)]
                        i_curr = idxs_curr.min() if idxs_curr.size > 0 else f_idx

                        mid = (i_prev + i_curr) // 2
                        labels[prev_frame:mid] = prev_id
                        labels[mid:f_idx] = curr_id

                    prev_id = curr_id
                    prev_frame = f_idx

            if prev_id >= 0:
                labels[prev_frame:] = prev_id

        Cid2[k] = labels

    return Cid2

def get_id_of_2dtrack(config_path: str, result_dir: str) -> list[dict[int, np.ndarray]]:
    """
    For each camera, load alldata.json and apply 2D → 2D tracklet ID voting.
    Returns a list of dicts: Cid2d_per_cam[camera_id] = {bbox_id: array(n_frame)}.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cam_ids = cfg["camera_id"]
    n_cam = len(cam_ids)

    ##Load all alldata.json for each camera
    data_per_cam = []
    for cam_id in cam_ids:
        with open(os.path.join(result_dir, str(cam_id), "alldata.json"), "r") as f:
            data_per_cam.append(json.load(f))
    n_frame = len(data_per_cam[0])

    ##Filter out detections with low ID confidence or duplicates
    for cam_idx in range(n_cam):
        for f_idx in range(n_frame):
            detections = data_per_cam[cam_idx][f_idx]
            cnts = np.zeros(20, int)
            for det in detections:
                bbox_id, cid, score = det[0], det[6], det[7]
                if cid in {0, 2, 3, 5} and score > CID_THR:
                    cnts[cid] += 1
            duplicates = np.where(cnts > 1)[0]
            for dup in duplicates:
                for det in detections:
                    if det[6] == int(dup):
                        det[7] = 0.0  ##Zero out confidence if duplicate

    ##Build tracklet-level 2D ID sequences per camera
    Cid2d_list = []
    for cam_idx in range(n_cam):
        tracklet_ids = {}
        for f_idx in range(n_frame):
            for det in data_per_cam[cam_idx][f_idx]:
                bbox_id = det[0]
                if bbox_id not in tracklet_ids:
                    tracklet_ids[bbox_id] = -2 * np.ones(n_frame, dtype=int)
                cid = det[6] if det[6] in {0, 2, 3, 5} and det[7] > CID_THR else -1
                tracklet_ids[bbox_id][f_idx] = cid
        ##Smooth each tracklet’s ID sequence
        wsize = 24 * 5
        tracklet_ids = set_id_for_each_frame_of_2dtracklets(tracklet_ids, n_frame, wsize)
        Cid2d_list.append(tracklet_ids)

    return Cid2d_list

##MAIN ENTRYPOINT

def proc(
    data_name: str,
    result_dir_root: str,
    raw_data_dir: str,
    config_path: str,
    show_result: bool = False,
):
    """
    Run keyframe matching over all cameras in a dataset.
    1. Load camera parameters.
    2. Load per-frame 2D detections and IDs.
    3. For every 12th frame, build info_dict and call MultiEstimator.predict_data().
    4. Save matched keyframes to match_keyframe.pickle.
    """
    result_dir = os.path.join(result_dir_root, data_name)
    camparam = get_camparam(config_path)

    ##Load per-camera alldata.json
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cam_ids = cfg["camera_id"]
    n_cam = len(cam_ids)

    T = []
    for cam_id in cam_ids:
        with open(os.path.join(result_dir, str(cam_id), "alldata.json"), "r") as f:
            T.append(json.load(f))

    ##Get 2D track IDs per camera
    Cid2d = get_id_of_2dtrack(config_path, result_dir)

    ##Load frame numbers per camera
    F = []
    for cam_id in cam_ids:
        frame_nums = np.load(os.path.join(result_dir, str(cam_id), "frame_num.npy"))
        F.append(frame_nums)

    ##Initialize MultiEstimator
    matcher = MultiEstimator(cfg=config_path)

    match_keyframes = []
    bcomb_prev: list = []

    ##Process every 12th frame between [1, n_frame-12)
    n_frame = len(T[0])
    for f_idx in tqdm(range(1, n_frame - 12, 12)):
        info_dict = {}

        for cam_idx in range(n_cam):
            ##If showing results, load raw image via imgstore
            if show_result:
                store_path = os.path.join(raw_data_dir, f"{data_name}.{cam_ids[cam_idx]}")
                store = imgstore.new_for_filename(os.path.join(store_path, "metadata.yaml"))
                img, _ = store.get_image(frame_number=F[cam_idx][f_idx])
            else:
                img = []

            per_frame = T[cam_idx][f_idx]
            entries = []
            for det in per_frame:
                bbox_id = [cam_idx, det[0]]
                bbox = det[1:5]
                cid = Cid2d[cam_idx][det[0]][f_idx]
                pose2d_raw = np.array(det[5])
                pose2d = undistort_points(config_path, cam_idx, pose2d_raw[:, :2])
                entries.append({
                    "pose2d": pose2d,
                    "pose2d_raw": pose2d_raw,
                    "bbox": bbox,
                    "bbox_id": bbox_id,
                    "cid": cid,
                })
            info_dict[cam_idx] = {0: entries, "image_data": img}

        matched_list, pose3d_list, bcomb = matcher.predict_data(
            info_dict, show=show_result, plt_id=0, camparam=camparam, bcomb_prev=bcomb_prev
        )
        bcomb_prev = bcomb

        if show_result:
            ##Example visualization on camera 0
            vis_cam = 0
            img_vis = info_dict[vis_cam]["image_data"]
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
            for pid, p3d in enumerate(pose3d_list):
                pts3d = p3d
                ##Add neck joint (midpoint of shoulders) for drawing
                neck = (pts3d[5] + pts3d[6]) / 2
                pts4 = np.vstack([pts3d, neck[np.newaxis, :]])
                reproj = reproject(vis_cam, pts4, camparam=camparam, config_path=config_path)
                kp_list = reproj.tolist()
                clean_kp(kp_list)
                draw_kps(img_vis, kp_list, mrksize=3, clr=colors[pid % len(colors)])
            img_res = cv2.resize(img_vis, (640, 480))
            cv2.imshow("Reprojection", img_res)
            cv2.waitKey(1)

        match_keyframes.append({
            "frame": f_idx,
            "bcomb": bcomb,
            "pose3d": pose3d_list
        })

    ##Save matches
    with open(os.path.join(result_dir, "match_keyframe.pickle"), "wb") as f:
        pickle.dump(match_keyframes, f)

if __name__ == "__main__":
    ##No-op when imported
    pass
