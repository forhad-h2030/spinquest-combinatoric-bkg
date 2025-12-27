# spinquest-combinatoric-bkg

Binary DNN classifiers for separating **J/ψ**, **ψ′**, **Drell–Yan**, and **combinatoric background** dimuon events in SpinQuest analyses.

This repository provides a **ROOT → NumPy preprocessing pipeline**, **binary DNN training**, and **application of trained models to experimental ROOT data**, with full support for **Rivanna (Slurm + GPU)**.

---
## Repository Structure

```text
spinquest-combinatoric-bkg/
├── preprocess/              # ROOT → NumPy feature extraction
│   ├── setup.sh             # ROOT + environment setup (Rivanna)
│   ├── run_extract.py       # Single-file wrapper (calls extract_dimu_features.py)
│   └── run_extract_all.py   # Convert ALL configured ROOT files (recommended)
│
├── scripts/                 # Training & inference scripts
│   ├── train_jpsi_vs_nonjpsi.py
│   ├── train_psip_vs_nonpsip.py
│   ├── train_dy_vs_comb.py
│   └── tag_exp_root_with_ml.py   # Apply trained models to experimental ROOT data
│
├── utils/                   # Core ML + plotting utilities
│   ├── core_train_binary.py      # Model definition, training loop, data handling
│   ├── extract_dimu_features.py  # Low-level ROOT → feature extractor
│   ├── plots_binary.py           # Diagnostic and performance plots
│   └── features.py               # Feature name definitions
│
├── submit_train_3.sh        # Slurm job submission script for three binary DNN trainings (Rivanna)
└── data/                    # User-provided ROOT files & generated NumPy inputs
```
## Running Instructions

### 1. Preprocess ROOT files → NumPy features
The DNN models do **not** read ROOT files directly. All ROOT inputs must be converted
into NumPy feature arrays first.

On Rivanna (or any system with ROOT available):

```bash
cd preprocess
source setup.sh              # set up ROOT + environment
python3 run_extract_all.py   # convert all configured ROOT files (recommended)
```
Skip this step if you already have the processed data (in npy format) or if you have access to the UVA Rivanna HPC (data location: /project/ptgroup/Forhad/spinquest-combinatoric-bkg/data).

### 2. Test training (short run on Rivanna)
Before submitting long GPU jobs, it is recommended to test the training workflow
on Rivanna with a **small number of epochs**.

Edit the training configuration in any of the training scripts
(e.g. `train_jpsi_vs_nonjpsi.py`, `train_psip_vs_nonpsip.py`, or `train_dy_vs_comb.py`):

```python
CFG = TrainConfig(
    epochs=3,          # short test run
    lr=5e-4,
    batch_size=1024,
    seed=42,
    standardize=False,
)
```

### 3. Full training on Rivanna (GPU, Slurm)
Train all three binary classifiers simultaneously using a Slurm job array:
```sbatch submit_train_3.sh
```
### 4. Apply trained models to experimental data
Once training is complete, tag experimental ROOT files with ML outputs:
```cd scripts
python3 tag_exp_root_with_ml.py
```
