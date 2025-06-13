from src.pipeline import step1_proc2d as step1

#included to run pictorial from noexec file path (nas_siddharth)
import importlib, types, sys

pictorial_mod = importlib.import_module("pictorial")
src_pkg = sys.modules.setdefault("src", types.ModuleType("src"))
m_lib_pkg = types.ModuleType("src.m_lib")
m_lib_pkg.__dict__.update(pictorial_mod.__dict__)
sys.modules["src.m_lib"] = m_lib_pkg
sys.modules["src.m_lib.pictorial"] = pictorial_mod
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from src.pipeline import step2_crossviewmatching as step2
from src.pipeline import step3_crossframematching as step3
from src.pipeline import step4_aniposefiltering as step4
from src.pipeline import visualize_result as vis

import imgstore

def proc(data_name, fps, results_dir_root, device_str, config_path,
         raw_data_dir, n_kp,vidfile_prefix=''):
    
    save_vid_at_step3 = False
    save_vid_cam = 6

    step1.proc(data_name, results_dir_root, raw_data_dir, device_str, fps)
    step2.proc(data_name,results_dir_root,raw_data_dir,config_path)  
    step3.proc(data_name,results_dir_root,raw_data_dir,config_path,save_vid_at_step3,save_vid_cam,vidfile_prefix=vidfile_prefix)     
    step4.proc(data_name,results_dir_root, config_path, n_kp, redo=True)   
    #vis.proc(data_name, save_vid_cam, config_path, raw_data_dir)
    #for 1-8
    import yaml
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ID = cfg['camera_id']
    for i_cam in range(len(ID)):
        print(f"[INFO] Generating video for camera {ID[i_cam]} (index {i_cam}) ...")
        vis.proc(data_name, i_cam, config_path, raw_data_dir)

       
if __name__ == '__main__':

    device_str='cuda:0'
    fps = 24
    n_kp = 17

    results_dir_root='./results3D'
    config_path = './calib/config.yaml' 

    raw_data_dir= './videos'
    data_name = 'example'


    proc(data_name, fps, results_dir_root, device_str, config_path, raw_data_dir, n_kp)




