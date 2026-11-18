# Calibration-Guided Thermal-Anomaly Detection for Reliable UAV Infrared Inspection of Photovoltaic Plants — Code & Annotation Archive

Companion archive for the manuscript submitted to *Transactions of the Institute of Measurement and Control* (TIMC). This repository deposits the analysis code, bounding-box annotations, evaluation splits, calibration records and baseline results needed to reproduce every reported number. The thermal imagery itself is **not** redistributed here: it is publicly available as the PVF-10 dataset (Wang et al., *Applied Energy* 376:124187, 2024); download it separately and place the 60×110 crops as described in `splits/split_record.txt`.

## Contents
- `code/` — proposal generator (`auto_pseudo*.py`), calibration & budget experiments (`exp__rev5_*`, `exp__rev6_*`, `exp__rev7_*`), own COCO-style evaluator (`exp__tim_eval.py`), attenuation/LODO/geometry experiments, baseline runner (`run_baselines_local.py`), statistics (`stats_phaseA/B.py`), figure script (`make_fig4.py`).
- `annotations/calibration_197/` — 197 manual calibration boxes (rater A), YOLO format.
- `annotations/test_350/` — frozen test-set ground truth (279 instances over 350 images).
- `annotations/agreement_sample50/` — stratified 50-image inter-annotator sample: `A/` (sealed reference), `B/` (independent human re-annotation), rules and manifest. Agreement: box-F1@IoU0.5 = 0.823, mean IoU 0.657, Jaccard 0.800 (`baselines/agreement_human.csv`).
- `splits/split_record.txt` — seed-42 split record (test 350 / train 1301 / val 143 / calibration pool 200).
- `calibration_records/` — frozen operating-point tuple (k=2.3, g=0.42, f_min=0.40, k_local=2.2, win=31, Δmin=3.0, τ=0.30, F2=0.6264), Table S2 weights, ±10% jitter study, headline provenance crosswalk.
- `baselines/` — matched-budget supervised references: in-house teacher–student reproduction (mAP@50 0.938) and consistency-screened published-recipe surrogate (0.932); AI-reader lower-bound probe.

## Reproduction
1. Obtain PVF-10 crops; reconstruct splits per `splits/split_record.txt`.
2. `python code/exp__rev5_f_budget.py` style entry points expect the original directory layout; adapt `BASE` paths in the scripts to your layout.
3. Evaluate with `code/exp__tim_eval.py` (own evaluator) or Ultralytics val (8.1.0) — both口径 are documented in `calibration_records/headline_provenance.csv`.
4. Environment: Python ≥3.10, torch 2.5+cu118, ultralytics 8.1.0, opencv-python, numpy, scipy, matplotlib.

## License
Code: MIT (see LICENSE). Annotations and records: CC-BY-4.0. Underlying imagery remains subject to the PVF-10 release terms.

## Citation
See CITATION.cff; the archived release DOI (Zenodo) will be inserted here upon minting.
