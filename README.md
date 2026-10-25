## Project Goal:
A platform where chest X-ray scans are uploaded. A classifier flags findings on chest X-rays, a segmentation model outlines structures in 3D scans, both with explanation heatmaps, reviewed by a radiologist

## Verified Environment:

- OS = Windows11 + WSL2(Ubuntu)
- Python = 3.14.4
- Project Disk (D:/) = 86.7 GB Free
- RAM = 8 GB Total(WSL Visible: 3.8Gi  Swap: 4.0Gi)
- GPU = Not Available (Kaggle Tesla T4 for Training)
![kaggle](./images/kaggle_gpu_check.png)


## Installation:

Execute the following commands inside your WSL terminal

 1. **Clone the Repository**:
 ```bash
 mkdir -p ~/projects
 cd ~/projects
 git clone https://github.com/gotham-saviour/medical_imaging_diagnosis.git
 cd medical_imaging_diagnosis
 ```

 2. **Set Up Virtual Environments**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Installing the Dependencies**

```bash
pip install -r requirements-dev.txt
```

4. **Install the Pre-Commit Git Hooks**

```bash
pre-commit install
```

5. **Run the Test Suite to verify Setup**

```bash
pytest
```

## **NOTICE**: For Education only, not Clinical Use
