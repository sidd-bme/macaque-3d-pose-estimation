# macaque3Dpose

<img src="./imgs/MovieS1.gif" width="800">

**macaque3Dpose** is a Python toolkit for markerless 3D pose estimation of multiple freely moving macaques, based on synchronized multi-view video input. The pipeline is designed for behavioral analysis in neuroscience, ethology, and animal sciences, but is generalizable to any group macaque environment with appropriate camera calibration.

---

## Overview
- **Multi-stage pipeline**: Detection (Swin-Mask R-CNN), Tracking (BoTSORT), Pose Estimation (ViTPose), ID Classification (ResNet152), and robust 3D keypoint triangulation and optimization.  
- **Pretrained model**: Outputs framewise 3D joint positions (eyes, nose, ears, shoulders, elbows, wrists, hips, knees, ankles) for all detected individuals.  
- **Modular and open**: Adaptable to new camera setups, models, and ID schemes.  
- **Research-ready**: Tested in the lab with group-housed Japanese macaques.  

---

## Getting Started
- For a **step-by-step demo**, see [getting_started.md](getting_started.md)  
- For **replication**, see [info_replication.md](info_replication.md)  
- For main code, see the [src/](./src/) directory and [run_demo.py](./run_demo.py)  

### Directory Structure
| Directory/file          | Purpose                                           |
|-------------------------|---------------------------------------------------|
| `src/`                  | Source code: pipeline, utils, and 3rd party       |
| `model/`                | Model configs for detection, pose, ID             |
| `configs/`              | TOML templates for camera and pipeline configs    |
| `notebooks/`            | Example and analysis Jupyter notebooks            |
| `calib/`                | Sample calibration folder                         |
| `videos/`               | Example minimal metadata for demo                 |
| `imgs/`                 | Sample output images/GIF for documentation        |
| `run_demo.py`           | Main pipeline runner                              |
| `getting_started.md`    | Beginner setup and quickstart                     |
| `info_replication.md`   | Detailed replication guide                        |
| `ThirdPartyNotices.txt` | Legal info on 3rd party code                      |

---

## Reference
If you use this code or dataset, please cite:

**Jumpei Matsumoto et al.**  
Three-dimensional markerless motion capture of multiple freely behaving monkeys toward automated characterization of social behavior. *Sci. Adv.* 11, eadn1355 (2025). DOI: [10.1126/sciadv.adn1355](https://doi.org/10.1126/sciadv.adn1355)

---

## Replicating or Extending the Pipeline
- Configuration templates are in [`configs/`](./configs/):  
  - [config_tmpl.toml](./configs/config_tmpl.toml) – main pipeline settings  
  - [calibration_tmpl.toml](./configs/calibration_tmpl.toml) – camera calibration template  

- Model config files in [`model/`](./model/):  
  - Detection: [Swin-Mask_R-CNN_bbox_only.py](./model/detection/SWIN-Mask_R-CNN_bbox_only.py)  
  - ID classifier: [sn_resnet152_8xb32_in1k_pretrained_optimized_finetuned.py](./model/id/sn_resnet152_8xb32_in1k_pretrained_optimized_finetuned.py)  
  - Pose: [td-hm_ViTPose-huge_8xb64-210e_coco-256x192_sn_macaque.py](./model/pose/td-hm_ViTPose-huge_8xb64-210e_coco-256x192_sn_macaque.py), [macaque.py](./model/pose/macaque.py)  

- Example minimal metadata: [`videos/example.22972495/metadata.yaml`](./videos/example.22972495/metadata.yaml)  
- Example demo/analysis notebooks: [`notebooks/`](./notebooks/)  

---

## License
- Distributed under the **Apache 2.0 License**.  
- Third-party components (e.g., anipose, mvpose) are included as permitted.  
  See [ThirdPartyNotices.txt](./ThirdPartyNotices.txt).  

---

## Acknowledgments
This work builds on previous major contributions:  
- [anipose](https://github.com/lambdaloop/anipose): Karashchuk et al., *Cell Reports*, 2021  
- [mvpose](https://github.com/zju3dv/mvpose): Dong et al., *arXiv* 2019  
- [openmmlab](https://github.com/open-mmlab): Chen et al., *arXiv* 2019  

**Funding:**  
Supported by MEXT/JSPS KAKENHI Grant Numbers 16H06534, 22H05157, 22K07325, 19H05467, 22K19480, 23H02781; JST Grant JPMJMS2295-12; Takeda Science Foundation; National Institutes of Natural Sciences Joint Grant 01111901.  

---

## Ethical Considerations
All animal experiments were approved by the Animal Welfare and Animal Care Committee of the Center for the Evolutionally Origins of the Human Behavior, Kyoto University, in accordance with institutional guidelines for nonhuman primates.  

---

## Data, Weights, and Results
- **Full model weights**: [Download Weights](https://drive.google.com/drive/folders/1_7SV-oecFph_s7XRUSj8mqeALtF50Qk0?usp=sharing)  
- **Sample metadata and pipeline configs** are included in this repo for demo/testing.  

---

## Contact
For academic inquiries or collaboration, open an issue or contact the authors as listed in the paper.
