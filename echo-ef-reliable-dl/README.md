# Clinically-Grounded, Quality-Conditioned Deep Learning with Conformal Uncertainty Calibration for Echocardiography

Code accompanying the paper *"Quality-Conditioned, Semi-Supervised Deep Learning with Conformal Uncertainty Calibration for Reliable Ejection Fraction Estimation and Heart-Failure Classification from Echocardiography."*

This repository contains the full, transparent experimental history described in the paper (Section 3.6): every notebook that produced a number reported in the paper, in the order it was run, including two identified-and-corrected implementation issues.

## Summary of results

| Stage | Notebook(s) | Input | Test EF MAE | Test HFrEF AUC |
|---|---|---|---|---|
| 1. Baseline | `02_baseline_supervised.ipynb`, `04_full_scale_training.ipynb` | Single frame | ~8.7 (val.) | ~0.83 (val.) |
| 2. Hybrid uncertainty | `10_hybrid_uncertainty_conformal_quality.ipynb` | Single frame | 8.06 | 0.810 |
| 3. Conformalized Quantile Regression | `11_conformalized_quantile_regression.ipynb` | Single frame | 7.30 | 0.807 |
| 4. Multi-frame ablation | `12_multiframe_extraction.ipynb` + `13_multiframe_final_model.ipynb` | 8 evenly-spaced frames | 6.36 | 0.911 |
| 5. **Final model** | `14_ed_es_extraction.ipynb` + `15_ed_es_final_model.ipynb` | 2 clinically-defined ED/ES keyframes | **6.05** | **0.921** |

A supplementary 2×2 component-ablation study (`16_component_ablation.ipynb`) isolating the individual contributions of the contraction-delta feature and the Quality-FiLM branch is also included; its results support the component-disentangling discussion added to the paper in response to reviewer/advisor feedback on Stage 4→5.

## Repository structure

```
notebooks/
  00_setup_and_eda.ipynb                     # Drive setup, initial EDA on EchoNet-Dynamic
  01_download_echonet.ipynb                  # Dataset download via Redivis API (Stanford AIMI)
  02_baseline_supervised.ipynb               # Dev-subsample baseline (Section 3, Stage 1)
  04_full_scale_training.ipynb               # Full-dataset baseline (Stage 1 numbers reported in paper)
  07_selfsupervised_pretraining.ipynb        # SimCLR contrastive pretraining (Section 3.4)
  08_finetune_from_pretrained.ipynb          # Label-efficiency sweep (Table 1)
  10_hybrid_uncertainty_conformal_quality.ipynb  # Stage 2: FiLM + heteroscedastic/epistemic uncertainty + Gaussian conformal
  11_conformalized_quantile_regression.ipynb # Stage 3: CQR replaces Gaussian conformal (single-frame)
  12_multiframe_extraction.ipynb             # 8-frame evenly-spaced extraction (ablation)
  13_multiframe_final_model.ipynb            # Stage 4: multi-frame ablation model + evaluation
  14_ed_es_extraction.ipynb                  # ED/ES keyframe extraction from expert tracings (Section 3.2)
  15_ed_es_final_model.ipynb                 # Stage 5 (FINAL): reported model in the paper
  16_component_ablation.ipynb                # Supplementary: 2x2 ablation of contraction-delta / Quality-FiLM
requirements.txt
README.md
```

## Reproducing the paper's results

1. **Get the data.** Register for a free Stanford AIMI account and accept the EchoNet-Dynamic Research Use Agreement, then run `01_download_echonet.ipynb` to pull the dataset via the Redivis API into your own Google Drive.
2. **Run the EDA** (`00_setup_and_eda.ipynb`) to confirm the data landed correctly and to reproduce Figure showing the EF/HFrEF class balance.
3. **Reproduce the final model** (what the paper reports): run, in order,
   `07_selfsupervised_pretraining.ipynb` → `14_ed_es_extraction.ipynb` → `15_ed_es_final_model.ipynb`.
   This reproduces Table 3's Stage 5 row and the paper's headline numbers (EF MAE 6.052, HFrEF AUC 0.921).
4. **Reproduce the full ablation history** (Table 3, all 5 stages): additionally run `02`/`04` (baseline), `10` (hybrid uncertainty), `11` (CQR), and `12`+`13` (multi-frame ablation).
5. **Reproduce the label-efficiency result** (Table 1): run `08_finetune_from_pretrained.ipynb` after `07`.
6. **Reproduce the component-ablation analysis** (supplementary, supports the Discussion's component-disentangling paragraph): run `16_component_ablation.ipynb` after `15`. This is a single-seed, first-pass disentangling exercise, flagged as such in the paper.

## Notes on the two corrected issues (Section 3.6 of the paper)

- `10_hybrid_uncertainty_conformal_quality.ipynb` originally shipped with a missing variance clamp at inference time, which is now fixed in the version in this repo (the clamp matches the one used during training). The commit history / notebook comments flag this explicitly.
- An earlier MC-Dropout-only uncertainty estimate (not included as a separate notebook here, superseded by `10`) produced an inverted, uninformative risk-coverage curve for EF regression; this motivated the heteroscedastic/CQR approach used from Stage 2 onward.

## Compute requirements

All notebooks are designed to run on **free-tier Google Colab** (a single T4 GPU) and free-tier Google Drive storage, with resumable, checkpointed training so a session disconnect never loses more than ~15 minutes of progress. See Section 3.7 of the paper for hyperparameters and total compute cost (a few GPU-hours for the final model).

## Data availability

The EchoNet-Dynamic dataset is not redistributed in this repository. It is available from Stanford AIMI via [Redivis](https://stanford.redivis.com), subject to a non-commercial research use agreement.

## Citation

If you use this code, please cite:

```
Barbod Yadali. "Quality-Conditioned, Semi-Supervised Deep Learning with Conformal Uncertainty
Calibration for Reliable Ejection Fraction Estimation and Heart-Failure Classification from
Echocardiography." Biomedical Signal Processing and Control, [year], [under review / DOI].
```

## License

MIT (see [LICENSE](LICENSE) for the code). Note: the EchoNet-Dynamic dataset itself has its own separate license/usage terms (Stanford AIMI / Redivis non-commercial research use agreement) and is not covered by this repository's license.
