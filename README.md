# Quality-Conditioned Deep Learning with Conformal Uncertainty Calibration for Echocardiography

Code accompanying the manuscript *"Quality-Conditioned, Semi-Supervised Deep Learning with Conformal Uncertainty Calibration for Reliable Ejection Fraction Estimation and Heart-Failure Classification from Echocardiography"* (submitted to *Biomedical Signal Processing and Control*).

The repository provides the full experimental pipeline as a sequence of Jupyter notebooks, designed to run end-to-end on **free-tier Google Colab and Google Drive**. All reported quantitative results are given in the manuscript; this repository contains the code needed to reproduce them.

## Overview

The pipeline estimates left-ventricular ejection fraction (EF) and classifies HFrEF (EF < 40%) from echocardiographic videos, with calibrated uncertainty. Its main components are:

- **Clinically-defined keyframes.** End-diastolic (ED) and end-systolic (ES) frames are identified from expert LV tracings via polygon-area (shoelace) computation.
- **Contraction-delta feature.** An explicit `f_ED - f_ES` feature is concatenated with the two frame embeddings, mirroring volume-difference computation in Simpson's method.
- **Quality-FiLM conditioning.** A learned feature-wise linear modulation branch conditions the backbone features on a hand-crafted image-quality score.
- **Self-supervised pretraining.** A SimCLR-pretrained ResNet-18 backbone (no EF labels) is used for initialization.
- **Conformalized Quantile Regression (CQR).** Distribution-free prediction intervals for EF with finite-sample coverage guarantees.
- **Selective prediction.** MC-Dropout predictive entropy with Clopper-Pearson lower confidence bounds for abstention on HFrEF classification.

## Repository structure

```
notebooks/
  00_setup_and_eda.ipynb                          # Drive setup, exploratory data analysis
  01_download_echonet.ipynb                       # Dataset download via Redivis (Stanford AIMI)
  02_baseline_supervised.ipynb                    # Stage 1: baseline on a development subsample
  04_full_scale_training.ipynb                    # Stage 1: baseline on the full dataset
  07_selfsupervised_pretraining.ipynb             # SimCLR contrastive pretraining
  08_finetune_from_pretrained.ipynb               # Label-efficiency evaluation (ImageNet vs. SimCLR init)
  10_hybrid_uncertainty_conformal_quality.ipynb   # Stage 2: FiLM + heteroscedastic/epistemic uncertainty + Gaussian conformal
  11_conformalized_quantile_regression.ipynb      # Stage 3: CQR (single frame)
  12_multiframe_extraction.ipynb                  # Stage 4: evenly-spaced multi-frame extraction
  13_multiframe_final_model.ipynb                 # Stage 4: multi-frame model and evaluation
  14_ed_es_extraction.ipynb                       # Stage 5: ED/ES keyframe extraction from expert tracings
  15_ed_es_final_model.ipynb                      # Stage 5: final model (ED/ES + delta + FiLM + CQR)
  16_component_ablation.ipynb                     # Supplementary: 2x2 ablation of contraction-delta and Quality-FiLM
requirements.txt
LICENSE
README.md
```

## Getting started

### 1. Data

The code uses the **EchoNet-Dynamic** dataset (Ouyang et al., *Nature* 2020), which is **not redistributed** here. Obtain access for free from Stanford AIMI via [Redivis](https://redivis.com/datasets/echonet-dynamic) by accepting the research use agreement. `01_download_echonet.ipynb` then downloads the data into your own Google Drive using Redivis's interactive login (no API token is stored in the notebook).

### 2. Environment

Notebooks target Google Colab (single T4 GPU). For a local environment:

```bash
pip install -r requirements.txt
```

### 3. Running the pipeline

Run the notebooks in numerical order. For the final model only:

```
01_download_echonet  ->  07_selfsupervised_pretraining  ->  14_ed_es_extraction  ->  15_ed_es_final_model
```

| Goal | Notebooks |
|---|---|
| Final model | `01` → `07` → `14` → `15` |
| Earlier stages (development history) | `02`, `04`, `10`, `11`, `12` + `13` |
| Label-efficiency evaluation | `07` → `08` |
| Component ablation | `15` → `16` |

## Design notes

- **Resumable and idempotent.** Preprocessing and training checkpoint to Google Drive (about every 15 minutes), so a Colab disconnect does not lose significant progress.
- **Official data split.** The official EchoNet-Dynamic train/validation/test split is used throughout. Model selection and design decisions use the validation set only.
- **Conformal calibration.** Calibration for CQR uses a held-out split of the validation data, separate from the test set.
- **Disclosed corrections.** `10_hybrid_uncertainty_conformal_quality.ipynb` contains a fix ensuring the log-variance clamp at inference matches the one used in training; this is noted in the notebook and discussed in the manuscript.

## Compute

All notebooks run on free-tier Colab with a few GPU-hours in total; hyperparameters are listed in the manuscript's Methods section.

## Citation

```
Yadali Jamalouei, B. et al. Quality-Conditioned, Semi-Supervised Deep Learning with Conformal Uncertainty
Calibration for Reliable Ejection Fraction Estimation and Heart-Failure Classification from
Echocardiography. Biomedical Signal Processing and Control (under review).
```

## License

Code is released under the MIT License (see [LICENSE](LICENSE)). The EchoNet-Dynamic dataset has its own separate terms of use and is not covered by this license.
