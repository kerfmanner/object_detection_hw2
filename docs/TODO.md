# TODO

Task list for the rest of the homework. I name the item to work on; the agent works only on that item and never starts the next one by itself. When the agent finishes an item, it ticks it and adds the commit or run that completed it. Items marked **(me)** are mine: the agent never does them (report prose, GPU runs, Kaggle, LMS).

Deadlines: Kaggle Mon Nov 10, 23:55 (choose 2 final submissions). Code and report on LMS Tue Nov 11, 23:55; no commits after that.

Details for every experiment: `docs/experiment_ideas.md`. Rubric items are quoted there per checkpoint.

## 0. Infrastructure

- [x] 0.1 Old-config check: `src.predict` and `src.evaluate` work on run directories created before `amp_dtype` / `grad_clip_norm` existed, with a test.
- [+] 0.2 **(me)** Sync runs to the laptop with `scripts/sync_runs.sh`;
- [+] 0.3 **(me)** Leaderboard metric line in AGENTS.md: COCO mAP@[.5:.95] (YOLO baseline val 0.267, LB 0.210).
- [+] 0.4 **(me)** Retry Kaggle phone verification once (mobile data). If it works: 0.5.
- [ ] 0.5 Kaggle launcher notebook (not committed): clone, install `requirements-kaggle.txt`, `DATA_ROOT`, W&B secret, `num_workers` 4, run `src.train`.

## 1. Two-stage: Faster R-CNN

- [x] 1.1 Configs and code: `lr_cosine` (cosine schedule, 10 epochs), `anchors_small` (sizes 16 to 256, 3 per location, 5 epochs), `giou_loss` (GIoU box loss, 5 epochs).
- [x] 1.2 **(me)** Run the three experiments; log rows; commit.
- [ ] 1.3 Optional, if time: RPN proposal counts or the 6-anchor variant.
- [x] 1.4 `notebooks/02_two_stage.ipynb` (template for 03 to 05): training setup, history, results table, experiments, inside the model (anchor coverage, RPN proposals, proposal recall), failure analysis with at least 3 cases, comparison. Shared tables and plots go to `src/visualization/`.
    - Done: notebook and helpers in ad00763, PC runs e1258ee and 416794b; final takeaways (§6.4, §7 numbers matched to the PC outputs) in the working tree, not committed. Conclusions (§9) are 1.5.
- [x] 1.5 write its conclusions; report notes on RPN vs second stage.
    - Done in the working tree, not committed: §9 of `02_two_stage.ipynb` (results, then hypotheses); its "What each stage contributes" bullet is the RPN vs second stage note. `report.md` is untouched (Hard Rule 2).

## 2. One-stage anchor-based: RetinaNet

- [x] 2.1 **(me)** Evaluate the baseline; log the row.
- [x] 2.2 Configs and code: `anchors_eda` (base sizes 16 to 256, 3 scales per octave, 5 epochs); `focal_loss` (configurable alpha and gamma, values with reasoning, 5 epochs). Done in the working tree, not committed yet.
- [x] 2.3 **(me)** Run both; log rows.
- [x] 2.4 `notebooks/03_one_stage_anchor.ipynb`: same template; inside the model (default vs EDA anchors, positive and negative anchors per image); removing the proposal stage; comparison with tuned Faster R-CNN.
    - Done in the working tree, not committed: notebook with takeaways and conclusions from a laptop run (no checkpoint needed); helpers `tabulate_anchor_matches`, `evaluate_frame_groups`. To run on the PC.
- [x] 2.5 **(me)** Run 03; conclusions.

## 3. Anchor-free: YOLO11

- [x] 3.1 Configs and code, 20 epochs each, one change vs the baseline: `imgsz_1280` (speed vs accuracy point); `tal_topk` (configurable TAL topk via a loss patch, topk 20 from the stride analysis); `scale_025` (scale augmentation 0.25 instead of 0.5). `p2_head` skipped for now. Done in the working tree, not committed yet.
- [x] 3.2 **(me)** Run both; log rows.
- [x] 3.3 `notebooks/04_anchor_free.ipynb`: same template; strides vs object sizes, location assignment; search range and selected settings table; comparison by object size with tuned anchor-based models; augmentation caveat.
    - Done in the working tree, not committed: notebook with takeaways and conclusions from a laptop run (no checkpoint needed); helper `count_assignment_candidates`. To run on the PC.
- [x] 3.4 **(me)** Run 04; conclusions.

## 4. Query-based: RT-DETR

- [x] 4.1 **(me)** Evaluate the baseline; log the row.
- [x] 4.2 Update the checkpoint TODOs in AGENTS.md and `docs/experiment_ideas.md` to R50 (only when I ask).
- [x] 4.3 Configs: two tuning experiments (choose from queries, matcher cost weights, denoising off, separate backbone LR), with reasoning. Chosen: `lr_scaled` (`lr` 2.5e-5 and `backbone_lr` 2.5e-6, linear scaling from batch 16 to 4, against the decline after best epoch 2); `denoising_groups` (`num_denoising` 200, so that dense frames get several denoising groups). Not chosen: more queries (recall at IoU 0.5 on val frames with 100 to 150 objects is already 0.90), matcher weights (no measured problem). Done in the working tree, not committed yet.
- [x] 4.4 **(me)** Run both; log rows.
- [x] 4.5 `notebooks/05_query_based.ipynb`: same template; top-query boxes on a dense frame; bipartite matching and no NMS; val mAP vs cumulative training time against Faster R-CNN; mAP by density and size; convergence statement.
    - Done in the working tree, not committed: notebook with takeaways and conclusions from a laptop run; helper `predict_query_boxes`. Left: the §6.3 takeaway (top-query boxes), which needs `rtdetr-baseline/best.pt` on the PC.
- [x] 4.6 **(me)** Run 05; conclusions.

## 5. Your pipeline

- [x] 5.1 Data and inference strategies (one change each, on the best model): augmentation for torchvision (flip, scale, brightness), ignored-region masking, step 2 frames, TTA or SAHI.
    - Code and configs in the working tree, not committed (2026-11-08): inference runs from an existing `best.pt` (`src.predict --inference-config`, `scripts/run_inference.sh`, `config/rtdetr/infer_*.toml`: I1 500 detections, I3 sizes 1152 / 1280 / 1408, flip member for I2); post-processing runs on saved predictions (`src/postprocess.py`, `scripts/run_postprocess.sh`, `config/postprocess/`): WBF and NMS fusion, temporal rescoring; `[augment]` and `[ignored_regions]` in the shared loop (`src/data/augment.py`; `config/rtdetr/augment_recipe.toml`, `augment_flip_photometric.toml`, `mask_ignored.toml`). Open: GPU runs (5.3), I4 tiling only if I3 helps small objects.
    - 2026-11-09, in the working tree, not committed: box smoothing along tracks (`[temporal]` `should_smooth_boxes`), weight EMA in the shared loop (`[ema]`, `config/rtdetr/ema.toml`), inference configs for the second queue (`docs/pipeline.md` §11.3). Measured so far: §11.1 and §11.2.
    - 2026-11-09, after the LB check (`docs/pipeline.md` §12.4): low-resolution augmentation (`[augment]` `low_res_prob`, `low_res_factor`) and half-resolution val inference (`[inference]` `input_scale`); `config/rtdetr/recipe_step2_low_res.toml` with `infer_recipe_step2_half`, `infer_recipe_step2_low_res_half`, `infer_recipe_step2_low_res_det500`. Chosen from test frame sizes, a deviation from data rule 4 decided by the user. Run on the PC (b4abfef to 229b014): fails its keep rules (half-resolution val +0.0058, ensemble D 0.3842 and 0.3843 against 0.3859), dropped; `docs/pipeline.md` §12.5.
    - 2026-11-09, in the working tree, not committed: `config/faster_rcnn/giou_recipe.toml` (the RT-DETR augmentation recipe on the Faster R-CNN member, `docs/pipeline.md` §12.9). Not run yet. Stacking prototype dropped (§12.9).
- [x] 5.2 Modelling: best tuned model trained longer, and a WBF ensemble of the best two or three families.
    - Idea: `yolo11m` as a modelling candidate (20.1M parameters vs 9.4M for yolo11s), so that the YOLO comparison is less confounded by model size; needs a timing run for batch size and epochs within the budget first.
    - WBF ensemble code and configs in the working tree, not committed (2026-11-08): `src/inference/fusion.py`, `config/postprocess/ens_*.toml`, tuned on saved val predictions. Open: the longer final RT-DETR (M1), whose settings wait for the D1 and D3 results.
    - 2026-11-09: M1 is `config/rtdetr/recipe_step2.toml` (D1 and D2 combined, 3 epochs), seed repeat `recipe_step2_repeat.toml`; post-processing candidates `config/postprocess/ens_recipe_*_smooth.toml`. Not run yet.
    - 2026-11-09, after the second queue: M1 val 0.3402 (seed 7: 0.3404), `infer_recipe_step2_det500` 0.3417. Final ensemble `config/postprocess/ens_rtdetr4_frcnn2_yolo2_smooth.toml` (8 members) 0.3859 on saved val predictions, not run on the PC yet; selection and submission plan in `docs/pipeline.md` §12.
    - 2026-11-09, train and val support: `train_splits` in the shared-loop baselines (val clips at the `train_frames` step, no monitoring, `best.pt` the last epoch); configs `config/rtdetr/trainval_recipe_step2.toml`, `trainval_recipe_step2_repeat.toml`, `trainval_recipe_step5.toml`, `infer_trainval_*`, `config/faster_rcnn/trainval_giou_loss.toml`, `trainval_baseline_2ep.toml`, `config/postprocess/ens_trainval_rtdetr4_frcnn2_yolo2_smooth.toml`. YOLO not supported (best epochs 7 and 2 of 20 under a decaying LR). Run on the PC (a102104 to 8ed1efe): train losses match the reference runs, val rows not comparable, LB not submitted yet; `docs/pipeline.md` §12.6 and §12.7.
- [ ] 5.3 **(me)** Run them; log rows.
- [x] 5.4 3-fold `GroupKFold` folds in `config/data.toml` and fold training for the final candidates.
    - 2026-11-09, in the working tree, not committed: `[folds]` (shuffled `GroupKFold` by clip, seed 42, 21 clips each), `clip_split` in every baseline, `load_clip_split` used by training, prediction, evaluation, and post-processing; configs `config/rtdetr/cv_recipe_step2_fold<k>.toml`, `config/faster_rcnn/cv_giou_loss_fold<k>.toml`, `config/postprocess/cv_ens_rtdetr_frcnn_smooth_fold<k>.toml`; `docs/pipeline.md` §12.8. Not run yet (5.5).
- [ ] 5.5 **(me)** Run the folds for the candidates.
- [ ] 5.6 `notebooks/06_final_pipeline.ipynb`: summary of all models; data and inference part; modelling part; table separating their effects; CV results; final selection; reproduction commands.
- [ ] 5.7 **(me)** Choose 2 final Kaggle submissions by val before Mon 23:55.

## 6. Report and submission (me)

- [ ] 6.1 **(me)** Data and validation design (split swap, frame step, why CV only for final candidates).
- [ ] 6.2 **(me)** Each family: setup, experiments including unsuccessful ones, metrics, public LB.
- [ ] 6.3 **(me)** Final pipeline and exact run instructions (dependencies, configs, seeds).
- [ ] 6.4 **(me)** References from `docs/literature.md`; figures in `images/`.
- [ ] 6.5 **(me)** Rerun all notebooks on the PC; check no errors or long printouts.
- [ ] 6.6 **(me)** Repository public, GitHub Pages deployed, link on LMS by Tue 23:55.

## Bonus (only if everything above is done)

- [x] B.1 Temporal post-processing (IoU linking and rescoring).
    - 2026-11-10, in the working tree, not committed: causal variant (`[temporal]` `is_causal`, `rescore_tracks` in `src/inference/temporal.py`) and `notebooks/07_bonus_temporality.ipynb` (motion between frames, ablation, error analysis, example track, transfer to other models and ensemble D, LB, cost), executed on the laptop from saved val predictions.
- [ ] B.2 Visualization (W&B project public; embeddings, attention maps or Grad-CAM).
    - 2026-11-10, in the working tree, not committed: `predict_query_internals`, `summarize_query_attention` (`src/models/rtdetr.py`), `plot_query_attention`, `plot_embedding_panels`; `notebooks/07_bonus_visualization.ipynb` (t-SNE of decoder query embeddings, deformable cross-attention per query and layer, statistics by size and density) for `rtdetr-recipe_step2`. Checked end to end on the laptop with COCO weights on 9 frames; to run on the PC (needs `best.pt`). W&B project public: (me).
