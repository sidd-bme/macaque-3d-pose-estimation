import os
from unittest import result
from src.utils import multicam_toolbox as mct

import copy
import cv2
import numpy as np
from tqdm import tqdm
from math import floor, ceil 
import yaml
import matplotlib.pyplot as plt
import math
import h5py
import pickle
import imgstore
import scipy.io
import glob

def ellipse_line(img, x1, x2, mrksize, clr):
    if x2[0] - x1[0] == 0:
        ang = 90
    else:
        ang = math.atan((x2[1] - x1[1])/(x2[0] - x1[0])) / math.pi * 180
    cen = ((x1[0] + x2[0])/2, (x1[1] + x2[1])/2)
    d = math.sqrt(math.pow(x2[0] - x1[0], 2) + math.pow(x2[1] - x1[1], 2) )
    
    cv2.ellipse(img, (cen, (d, mrksize), ang), clr, thickness=-1)
    return 0

def clean_kp(kp, ignore_score=False, show_as_possible=True):

    cnt = 0
    for i_kp in range(len(kp)):
        if kp[i_kp][2] > 0.3:
            cnt += 1

    for i_kp in range(len(kp)):

        if i_kp == 1 or i_kp == 2:
            kp[i_kp] = None
            continue

        if show_as_possible:
            if cnt == 0:
                kp[i_kp] = None
            elif np.isnan(kp[i_kp][0]):
                kp[i_kp] = None
            elif kp[i_kp][0] > 3000 or kp[i_kp][0] < -1000 or kp[i_kp][1] > 3000 or kp[i_kp][1] < -1000:
                kp[i_kp] = None
            else:
                kp[i_kp] = kp[i_kp][0:2]
        else:
            if kp[i_kp][2] < 0.3 and not ignore_score:
                kp[i_kp] = None
            elif np.isnan(kp[i_kp][2]) and not ignore_score:
                kp[i_kp] = None
            elif np.isnan(kp[i_kp][0]):
                kp[i_kp] = None
            elif kp[i_kp][0] > 3000 or kp[i_kp][0] < -1000 or kp[i_kp][1] > 3000 or kp[i_kp][1] < -1000:
                kp[i_kp] = None
            else:
                kp[i_kp] = kp[i_kp][0:2]

def add_neckkp(kp):
    if kp[5] is not None and kp[6]is not None:
        d = [(kp[5][0]+kp[6][0])/2, (kp[5][1]+kp[6][1])/2]
    else:
        d = None
    kp.append(d)

def draw_kps(img, kp, mrksize, clr=None):

    #kp_con = [
    #    #(0,2), # nose - right eye
    #    #(0,1), # nose - left eye
    #    (0,3), # nose - left ear
    #    (0,4), # nose - right ear
    #    #(2,4), # right eye - right ear
    #    #(1,3), # left eye - left ear
    #    #(0,4), # nose - right ear
    #    #(0,3), # nose - left ear
    #    (6,8), # right shoulder - right elbow
    #    (5,7), # left shoulder - left elbow
    #    (8,10), # right elbow - right wrist
    #    (7,9), # left elbow - left wrist
    #    (12,14), # right hip - right knee
    #    (11,13), # left hip - left knee
    #    (14,16), # right knee - right ankle
    #    (13,15), # left knee - left ankle
    #    (0,17), # nose - neck
    #    (17,6), # neck - right shoulder
    #    (17,5), # neck - left shoulder
    #    (17,12), # neck - right hip
    #    (17,11) # neck - left hip
    #]

    kp_con = [
    (1, 2),
    (0, 1),   # nose - left eye
    (0, 2),   # nose - right eye
    (1, 3),   # left eye - left ear
    (2, 4),   # right eye - right ear
    (3, 4),   # left ear - right ear
    (0, 17),  # nose - neck
    (3, 17),  # left ear - neck
    (4, 17),  # right ear - neck
    (17, 5),  # neck - left shoulder
    (17, 6),  # neck - right shoulder
    (5, 6),   # left shoulder - right shoulder
    (17, 11), # neck - left hip
    (17, 12), # neck - right hip
    (11, 12), # left hip - right hip
    (5, 7),   # left shoulder - left elbow
    (7, 9),   # left elbow - left wrist
    (6, 8),   # right shoulder - right elbow
    (8, 10),  # right elbow - right wrist
    (11, 13), # left hip - left knee
    (13, 15), # left knee - left ankle
    (12, 14), # right hip - right knee
    (14, 16), # right knee - right ankle
    # Diagonals for torso
    (5, 11),  # left shoulder - left hip
    (6, 12),  # right shoulder - right hip
    (5, 12),  # left shoulder - right hip
    (6, 11),  # right shoulder - left hip
    ]


    if clr is not None:
        c = (int(clr[0]*255), int(clr[1]*255), int(clr[2]*255))
    else:
        c = (0, 0, 0)

    
    for idx in reversed(range(len(kp))):
        if kp[idx] is not None and idx not in (1, 2):
            cv2.circle(img, (int(kp[idx][0]), int(kp[idx][1])), mrksize, c, thickness=-1)

    for i1, i2 in kp_con:
        if kp[i1] is not None and kp[i2] is not None:
            ellipse_line(img, kp[i1], kp[i2], mrksize, c)

def reproject(config_path, i_cam, p3d):

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ID = cfg['camera_id']
    id = ID[i_cam]

    path_intrin = os.path.dirname(config_path) + '/cam_intrinsic.h5'
    path_extrin = os.path.dirname(config_path) + '/cam_extrinsic_optim.h5'

    with h5py.File(path_intrin, mode='r') as f_intrin:
        mtx = f_intrin['/'+str(id)+'/mtx'][()]
        dist = f_intrin['/'+str(id)+'/dist'][()]
        K = f_intrin['/'+str(id)+'/K'][()]
        xi = f_intrin['/'+str(id)+'/xi'][()]
        D = f_intrin['/'+str(id)+'/D'][()]
    with h5py.File(path_extrin, mode='r') as f_extrin:
        rvecs = f_extrin['/'+str(id)+'/rvec'][()]
        tvecs = f_extrin['/'+str(id)+'/tvec'][()]

    pts, _ = cv2.omnidir.projectPoints(np.reshape(p3d, (-1,1,3)), rvecs, tvecs, K, xi[0][0], D)

    return pts[:,0,:]

def proc(data_name, i_cam, config_path, raw_data_dir=None):

    result_dir = './results3D/' + data_name

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ID = cfg['camera_id']

    vid_path = './output/' + data_name + '_{:d}.mp4'.format(ID[i_cam])

    if os.path.exists(vid_path):
        print('already exists:', vid_path)
        return

    if os.path.exists(result_dir + '/kp3d_fxdJointLen.pickle'):
        kp3d_file = result_dir + '/kp3d_fxdJointLen.pickle'
    elif os.path.exists(result_dir + '/kp3d.pickle'):
        kp3d_file = result_dir + '/kp3d.pickle'
    else:
        print('no kp3d file:', data_name)
        return
        

    print('generating...', vid_path)

    ignore_score = False
    show_as_possible = True

    #vw = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (800, 600)) 
    #increased op res
    vw = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (2048, 1536))
    with open(kp3d_file, 'rb') as f:
        data = pickle.load(f)

    data_name = os.path.basename(result_dir)
    mdata = raw_data_dir + '/' + data_name.split('.')[0] + '.' + str(ID[i_cam]) + '/metadata.yaml'
    frame_num = np.load(result_dir + '/' + str(ID[i_cam]) + '/frame_num.npy')
    store = imgstore.new_for_filename(mdata)

    meta = store.get_frame_metadata()
    store_frames = [int(f) for f in meta['frame_number']]
    store_frames_set = set(store_frames)

    #print(f"First 10 store_frames: {store_frames[:10]}")
    #print(f"Type of store_frames[0]: {type(store_frames[0])}")
    #print(f"First 10 frame_num: {frame_num[:10]}")
    #print(f"Type of frame_num[0]: {type(frame_num[0])}")


    X = data['kp3d']
    S = data['kp3d_score']

    n_animal, n_frame, n_kp, _ = X.shape
    n_cam = len(ID)

    clrs = [(1,0,0), (0,1,0), (0,0,1), (1,1,1)]

    frame_number = -1
    #changed for error handling for extra/missed frames^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #for i_frame in tqdm(range(n_frame)):

    #    if frame_number >= frame_num[i_frame]:
    #        pass
    #    if frame_number == -1:
    #        frame, (frame_number, frame_time) = store.get_image(frame_num[i_frame])
    #    else:
    #        while frame_number < frame_num[i_frame]:
    #            frame, (frame_number, frame_time) = store.get_next_image()
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    for i_frame in tqdm(range(n_frame)):
        fn = int(frame_num[i_frame])
        if i_frame < 5:
            print(f"Trying frame {fn} (type: {type(fn)}) - In store_frames_set? {fn in store_frames_set}")
        if fn not in store_frames_set:
            print(f'[warning] Skipping frame {fn}: not found in imgstore')
            continue
        try:
            frame, (frame_number, frame_time) = store.get_image(fn)
        except Exception as e:
            print(f'[warning] Error loading frame {fn}: {e}')
            continue
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        img = copy.deepcopy(frame)

        for i_animal in range(n_animal):
            x = X[i_animal, i_frame, :, :]
            s = S[i_animal, i_frame, :]

            a = (x[5,:] + x[6,:])/2
            x = np.concatenate([x,a[np.newaxis,:]],axis=0)

            a = (s[5] + s[6])/2
            s = np.concatenate([s,a[np.newaxis]],axis=0)

            p = reproject(config_path, i_cam, x)
            I = np.logical_not(x[:,0]==0)
            I = np.logical_and(I, s>0.0)
            p = np.concatenate([p, I[:,np.newaxis]], axis=1)

            kp = p.tolist()
            mrksize = 3
            clean_kp(kp, ignore_score=ignore_score, show_as_possible=show_as_possible)
            
            if i_animal < 4:
                draw_kps(img, kp, mrksize, clrs[i_animal])
            else: 
                draw_kps(img, kp, mrksize, (0,0,0))

        #cv2.putText(img, 'Frame:{:05d}'.format(i_frame), (int(30), int(50)), 
        #    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #    fontScale=1.5,
        #    color=(255,255,255), thickness=3)

        #img = cv2.resize(img, (800,600))
        #resize here too
        #img = cv2.resize(img, (2048,1536))
        vw.write(img)

    vw.release()

if __name__ == '__main__':

    pass



    
