# Tested environment 
Ubuntu 20.04.6, NVIDIA A5000 (NVIDIA Driver Version: 550.90.07)

# Python version 
3.9.21

# Installing tools
```bash
cd $PathForDownloadedFolder

# pytorch 
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# Openmmlab
pip install openmim==0.3.9
mim install mmengine==0.10.7
mim install mmcv==2.1.0 ##mmcv-full depreciated
mim install mmdet==3.2.0
mim install mmpose==1.3.2
pip install boxmot==12.0.7 ##refer https://github.com/mikel-brostrom/boxmot 
mim install mmpretrain==1.2.0 ##mmcls depreciated
pip install xtcocotools==1.14.3

# Major tools
pip install opencv-python==4.11.0.86
pip install opencv-contrib-python==4.11.0.86
pip install numba==0.60.0
pip install h5py==3.13.0
pip install pyyaml==6.0.2
pip install toml==0.10.2 
pip install matplotlib==3.8.4
pip install joblib==1.5.0
pip install networkx==3.2.1

# Minor or local resources 
pip install imgstore==0.3.7
cd src/m_lib/
python setup.py build_ext --inplace
```

# Run demo 
```bash 
cd $PathForDownloadedFolder

python run_demo.py # the results will appear at "./output"
```
