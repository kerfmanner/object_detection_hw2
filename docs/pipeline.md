# Final pipeline plan

Plan for TODO section 5 (Your pipeline). It collects what the runs so far show, the ideas already in `docs/experiment_ideas.md`, and ideas from the course reading list (`docs/literature.md`), ordered by expected value per GPU hour. It is a plan, not a list of tasks: each item is implemented only when named.

Deadlines: Kaggle Mon Nov 10, 23:55 (2 final submissions); code and report Tue Nov 11, 23:55.

## 1. Rubric and rules

- Push the leaderboard score with **data or inference strategies** and **modelling experiments**, graded separately; include intermediate comparisons that separate their effects.
- Final pipeline justified by earlier results, reproducible, with a summary comparison of all models.
- One change at a time against a fixed reference; full val is the primary metric; gains below the seed noise count only after a repeat with another seed.
- Tune on val only. The public LB is a final check for selected models. Test frames are never used to fit or choose anything.
- Competition data only; pretrained weights from ImageNet, COCO, or Objects365 only; ask before adding a dependency.

## 2. Where we stand (measured)

| Model | val mAP | AP50 | AP_small | val mAP ≤ 1920 px | public LB | ms/image |
| --- | --- | --- | --- | --- | --- | --- |
| `rtdetr-lr_scaled` | 0.3305 | 0.6140 | 0.1352 | 0.3079 | | 51.6 |
| `rtdetr-denoising_groups` | 0.3274 | 0.6171 | 0.1367 | 0.3012 | | 51.6 |
| `rtdetr-lr_scaled_repeat-s7` (seed 7) | 0.3211 | 0.6122 | 0.1293 | 0.2971 | | |
| `rtdetr-baseline` (reference, RT-DETRv2 R50) | 0.3209 | 0.6027 | 0.1334 | 0.2935 | 0.2563 | 53.2 |
| `faster_rcnn-giou_loss` (tuned two-stage) | 0.3066 | 0.6258 | 0.1203 | 0.2902 | | 60.8 |
| `faster_rcnn-baseline` | 0.3013 | 0.6159 | 0.1069 | 0.2819 | 0.2273 | 62.5 |
| `retinanet-baseline` | 0.2689 | 0.5116 | 0.0780 | 0.2631 | 0.1562 | 48.6 |
| `retinanet-focal_loss` (gamma 1) | 0.2671 | 0.5102 | 0.0807 | 0.2609 | | 47.3 |
| `yolo-baseline` | 0.2671 | 0.5242 | 0.1119 | 0.2483 | 0.2097 | 13.7 |
| `yolo-tal_topk` | 0.2671 | 0.5334 | 0.1146 | 0.2417 | | 13.3 |
| `yolo-scale_025` | 0.2667 | 0.5217 | 0.1094 | 0.2581 | | 12.7 |
| `yolo-imgsz_1280` | 0.2498 | 0.4963 | 0.1119 | 0.2411 | | 7.9 |
| `retinanet-anchors_eda` | 0.2271 | 0.4968 | 0.0766 | 0.2247 | | 48.0 |

Tuning so far, against each family baseline:

- RT-DETR: `lr_scaled` +0.0096 mAP at seed 42 but +0.0002 at seed 7 (`lr_scaled_repeat-s7`), so its gain is not established; `denoising_groups` +0.0065, below the seed spread of 0.0094. Both seed-42 gains come almost entirely from bus and truck (observation 7). In both seeds `lr_scaled` has higher AP50 (+0.0113, +0.0095) and AP_medium (+0.046, +0.023) and lower AP_large (−0.010, −0.016) than the baseline, and a milder decline by epoch 7 (monitoring 0.3045 and 0.3074 against 0.2960).
- Faster R-CNN: `giou_loss` +0.0053, `lr_cosine` +0.0049, `anchors_small` +0.0024; none established beyond seed noise (`02_two_stage.ipynb` §5).
- RetinaNet: `focal_loss` (gamma 1) −0.0018, identical at epoch 4 (0.2675 vs 0.2676); `anchors_eda` −0.0418, with AP75 −0.078 and AP_large −0.087 and no AP_small gain.
- YOLO: `tal_topk` and `scale_025` within 0.0004 of the baseline; `imgsz_1280` −0.0173 at 1.7 times the speed.

Observations that drive the plan:

1. **Early peak, then decline.** RT-DETR peaks at epoch 2 of 7 (monitoring mAP 0.323, then 0.296 at epoch 7) and Faster R-CNN at epoch 2 of 10, while every training loss keeps falling. A 4 times lower LR (`lr_scaled`) raises the peak (0.3315 at epoch 2) but keeps the decline (0.3045 at epoch 7). No family uses augmentation (torchvision and Hugging Face loops). Hypothesis: overfitting to 56 clips; more varied training data should help more than more epochs or a lower LR.
2. **Small objects are the main loss.** Tuned Faster R-CNN detects 66.4% of small val boxes at any score (97.0% of large), and misses are 34% of GT at the best-F1 threshold (`02_two_stage.ipynb` §7). 96% of objects are pooled from P2. Smaller anchors did not help in either anchor-based family (Faster R-CNN +0.0038 AP_small; RetinaNet −0.0014). The RPN covers 72.4% of small boxes with a proposal at IoU 0.5, but only 64.8% are detected (§6.3). Hypothesis: input resolution and the second stage, not anchors, limit small objects.
3. **RT-DETR is capped at 300 detections per frame** (all val frames hit the cap), while the other families keep 500, the LB limit.
4. **The families make different errors.** Faster R-CNN has the best AP50, RT-DETR the best AP75 and AP_small, and YOLO is 4 times faster. This is the usual condition for an ensemble to help.
5. **Val vs LB.** The ≤ 1920 px val subset ranks the four baselines as the LB does, except RetinaNet (0.263 val subset, 0.156 LB). LB is 0.06 to 0.11 below full val. Treat the ≤ 1920 px number as the closer proxy, but select on full val as agreed.
6. **Dark frames.** 13.2% of sampled val frames are dark (mean luminance < 0.25) against 2.1% of train (`01_eda.ipynb` §12.3).
7. **Per-class noise.** The RT-DETR tuning gains come from bus and truck: `lr_scaled` bus +0.087 and truck +0.034 AP, the other six classes −0.006 to −0.013; `denoising_groups` truck +0.067 and bus +0.022, van −0.039. Bus has 2 val clips and 6 tracks, truck 15 tracks (`01_eda.ipynb` §9.2, §11.3), so one clip can move mAP by about 0.01. Measured: `lr_scaled` at seeds 42 and 7 differs by 0.0094 mAP (0.0108 on ≤ 1920 px, 0.0059 AP_small), with bus 0.530 against 0.458 and truck 0.279 against 0.251, ten times the Faster R-CNN seed difference (0.0009). For RT-DETR, a change counts only if it beats 0.0094 or holds across seeds. Every pipeline step should report per-class AP next to mAP.
8. **AP follows object size, not class frequency.** Bus is 0.8% of train boxes and has the second-best AP (0.443 RT-DETR, median √area 106.6 px); motor is 9.2% and has the worst (0.161, 40.2 px). Focal loss gamma made no difference in RetinaNet (observation in the tuning list above).
9. **Ignored regions are trained as background.** `VisDroneDataset` returns `ignored_boxes`, but no training loop uses them, so anchors, RoIs, and queries on crowds and dense parking train as background (72 to 75% of frames have such regions, `01_eda.ipynb` §7). The tuned Faster R-CNN still puts 5,517 of its detections above the best-F1 threshold (5.5%) at least half inside ignored regions, 4,136 of them persons, with median score 0.60 (correct detections 0.91).

Reference model for the pipeline: `rtdetr-baseline`. The seed repeat of `lr_scaled` fell back to the baseline's level (0.3211 against 0.3209), so neither RT-DETR tuning run is established. `lr_scaled` remains a candidate for M1 because of its milder decline over epochs (observation 1), which matters for a longer schedule.

## 3. Inference strategies (no training)

Each needs only `best.pt` and a val prediction pass (about 3 min per pass for RT-DETR on the PC). These are the cheapest experiments and should run first.

| ID | Idea | Evidence or source | Code | GPU cost |
| --- | --- | --- | --- | --- |
| I1 | RT-DETR keeps up to 500 (query, class) pairs instead of 300 | Observation 3; the official post-processor ranks all query and class pairs ([RT-DETR code](https://github.com/lyuwenyu/RT-DETR)) | relax the `detections_per_img ≤ num_queries` check | 1 val pass |
| I2 | Horizontal-flip TTA merged with WBF | [Weighted Boxes Fusion](https://arxiv.org/abs/1910.13302); [Bag of Freebies](https://arxiv.org/abs/1902.04103) | flip pass + WBF (implement from the paper, no new dependency) | 2 val passes |
| I3 | Larger test resolution (shorter side 1280 instead of 1024) | Observation 2; [small object survey](https://arxiv.org/abs/2207.14096) (multi-scale inference) | config only for torchvision and RT-DETR | 1 val pass, about 1.5 times slower |
| I4 | SAHI-style tiling: 2 × 2 overlapping tiles plus the full frame, merged per class | [SAHI](https://arxiv.org/abs/2202.06934); observation 2 | tiling and merge (implement, or ask to add `sahi`) | about 5 times inference |
| I5 | Soft-NMS for Faster R-CNN | [Soft-NMS](https://arxiv.org/abs/1704.04503) | replace the final NMS | 1 val pass |
| I6 | Temporal rescoring: link boxes across consecutive frames by IoU and raise the scores of consistent tracks (Seq-NMS style); also the bonus B.1 | [Seq-NMS](https://arxiv.org/abs/1602.08465), [T-CNN](https://arxiv.org/abs/1604.02532), [ByteTrack](https://arxiv.org/abs/2110.06864) (keep low-score boxes on tracks); same object IoU at k = 1 median 0.868 on val (`01_eda.ipynb` §9) | post-processing on prediction tables | none: runs on saved predictions on the laptop |

Expected value, as hypotheses: I1 is small but almost free. I3 and I4 target the biggest error (small objects). I5 is low priority, because duplicates are only 240 detections at the best-F1 threshold for Faster R-CNN. I6 is free to tune and also earns the bonus.

## 4. Data strategies (training)

One training run each on the reference model, at the tuning budget (step 5, about 1.5 GPU hours).

| ID | Idea | Evidence or source | Code |
| --- | --- | --- | --- |
| D1 | Augmentation in the shared torchvision and RT-DETR loop: horizontal flip, random scale (multi-scale training), photometric distortion | Observations 1 and 6; the official RT-DETRv2 recipe trains with photometric distortion (p 0.5), zoom-out, IoU crop (p 0.8), horizontal flip, and 13 input scales from 480 to 800 px, and switches them off for the last epochs ([dataloader config](https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetrv2_pytorch/configs/rtdetrv2/include/dataloader.yml)); [YOLOv4](https://arxiv.org/abs/2004.10934) | box-aware transforms in `src/data/` with config flags |
| D2 | Train frames at step 2 instead of 5 (12,240 frames) | Observation 1; 2.5 times more frames, and at step 5 40% of box pairs of the same object already have IoU < 0.5 (`01_eda.ipynb` §9.1), so the added frames show new positions | config only; for RT-DETR already prepared as `config/rtdetr/step2.toml` (`rtdetr-step2`, 3 epochs at the 1.5 hour budget) and run on the PC while the pipeline code is written |
| D3 | Ignored-region masking: fill ignored regions with the train mean colour, except inside labelled boxes | AGENTS.md (final-pipeline experiment); observation 9 | masking in the dataset; works for every family |
| D4 | Rare-class copy-paste (box-only) | [Simple Copy-Paste](https://arxiv.org/abs/2012.07177); bus has 2 val clips and 6 tracks, truck 15 tracks (`01_eda.ipynb` §9.2, §11.3) | new augmentation; expensive to write |

Suggested split of D1 into two runs if time allows: flip plus photometric first (cheap, no box geometry risk), then multi-scale. If time is short, run D1 as one "recipe augmentation" factor and say so in the notes.

## 5. Modelling

| ID | Idea | Evidence or source | Cost |
| --- | --- | --- | --- |
| M1 | Final RT-DETR: best tuning settings plus the winning data strategy, step 2, longer schedule with an LR drop or EMA | Observation 1; RT-DETRv2 recipe uses EMA (decay 0.9999) | EMA needs code; about 3 to 5 GPU hours |
| M2 | WBF ensemble of RT-DETR, tuned Faster R-CNN, and YOLO, with per-model weights tuned on val | [Weighted Boxes Fusion](https://arxiv.org/abs/1910.13302); observation 4 | WBF code (shared with I2); fusion runs on saved predictions on the laptop |
| M3 | Larger or newer query-based model: RT-DETRv2 R101, or D-FINE / DEIM (COCO or Objects365 weights) | [D-FINE](https://arxiv.org/abs/2410.13842), [DEIM](https://arxiv.org/abs/2412.04234): faster convergence and better box regression | timing run first; risky before Monday |
| M4 | `yolo11m` (already noted under TODO 5.2) | larger model, fair size comparison | timing run first |
| M5 | NWD for tiny boxes in matching or loss | [Normalized Wasserstein Distance](https://arxiv.org/abs/2110.13389); observation 2 | loss and matcher changes; stretch |
| M6 | Cascade-style refinement for localization | [Cascade R-CNN](https://arxiv.org/abs/1712.00726) | not in torchvision; out of scope for this deadline |
| M7 | Ignore-aware loss: anchors and RoIs at least half inside an ignored region (the evaluator's rule) become "don't care" (label −1, skipped by torchvision's sampler) instead of background; for RT-DETR, mask the no-object loss of unmatched queries inside ignored regions | Observation 9; [VisDrone](https://arxiv.org/abs/2001.06303) (ignored regions are not evaluated) | Faster R-CNN about 60 lines like `GIoURoIHeads`; RT-DETR needs a change inside the Hugging Face loss, riskier before Monday |

D3 and M7 are two routes to the same problem: D3 removes the regions from the input for every family, M7 keeps the input and removes only the wrong labels. With the time left, D3 on the reference RT-DETR comes first; M7 for Faster R-CNN is the next option and would also strengthen the ensemble.

M2 is the strongest modelling candidate for the time left. It reuses runs that already exist, and the fusion can be tuned on the laptop from `val_predictions.parquet` and applied to each run's test `predictions.csv`.

## 6. Order of work (proposal)

| When | GPU (PC) | Laptop |
| --- | --- | --- |
| Sat night | queued runs (done, section 2) | |
| Sun morning | `lr_scaled_repeat` (seed 7, done: no gain); notebook 02 rerun (done) | inference code for I1, I3, I2 |
| Sun day | I1, I3, I2 on `rtdetr-baseline` (val passes only); D1 (augmentation); D3 (masking) | WBF and temporal rescoring (I2, I6, M2), tuned on saved val predictions; `06_final_pipeline.ipynb` skeleton |
| Sun night | M1: final RT-DETR with the winning settings at step 2 | |
| Mon morning | test predictions; I4 if time allows; seed repeat or 3-fold check for the final candidate | ensemble of final models (M2), effect table |
| Mon by 23:55 | | choose 2 submissions by val |

3-fold `GroupKFold` (TODO 5.4) costs 3 trainings per candidate. With the time left, it fits one candidate at most, probably at step 5.

## 7. Selecting the two final submissions

1. The best single pipeline by full val mAP: final RT-DETR plus the inference strategies that helped.
2. The best ensemble by full val mAP (M2), if it beats submission 1 beyond noise; otherwise the second-best single pipeline.

The LB score of each is recorded in `results.csv` (`public_lb`) but does not change the choice.

## 8. Not planned

- External or extra VisDrone data, VisDrone-trained checkpoints, pseudo-labels on test, statistics from test frames (rules).
- MMDetection models (AGENTS.md: avoid `mmcv`).
- Focal loss or other class-imbalance losses: focal loss already acts on object versus background imbalance (RetinaNet, the varifocal loss of RT-DETR), its gamma made no measured difference (`retinanet-focal_loss`), and AP follows object size rather than class frequency (observation 8). Most errors are misses and background, not confusions between rare and frequent classes (`02_two_stage.ipynb` §7.2).
- Feature-level video methods ([FGFA](https://arxiv.org/abs/1703.10025), [SELSA](https://arxiv.org/abs/1907.06390), [MEGA](https://arxiv.org/abs/2003.00586)) and [QueryDet](https://arxiv.org/abs/2106.09018) / [ClusDet](https://arxiv.org/abs/1904.08008): too much code for the time left; I4 and I6 cover the same needs more cheaply.

## 9. Recommendations

- The reference is settled: `rtdetr-baseline` (the `lr_scaled` gain did not repeat). Judge RT-DETR changes against the seed spread of 0.0094 mAP and by per-class AP.
- Run the inference experiments next. They take minutes of GPU time on existing checkpoints.
- Build the ensemble and temporal rescoring on the laptop in parallel. Both work on saved predictions, so they need no GPU.
- Make augmentation the main data experiment. It is the most likely fix for the early peak.
- Handle ignored regions in training (D3, then M7) instead of class-imbalance losses (section 8).
- Expect cross-validation to fit only one final candidate, given the time left.
## 10. Pipeline status (2026-11-08, phase A)

### 10.1 Built (laptop, tested)

- **Inference runs** (`src.predict --inference-config`, `scripts/run_inference.sh`): an `config/<family>/infer_<name>.toml` names a `source` experiment and changes only inference settings (`[inference]`, `min_size`, `max_size`, `imgsz`). The new run `runs/<family>-infer_<name>/<timestamp>/` gets the merged config, a relative `best.pt` symlink to the source checkpoint, and a `metrics.json` with the source's training record plus `source_run`, `source_config`, `source_commit`. `scripts/finish_run.sh` and `src.evaluate` work on it unchanged, so each inference experiment gets its own `results.csv` row.
- **Post-processing runs** (`src/postprocess.py`, `scripts/run_postprocess.sh`, `config/postprocess/`): fuse (WBF or NMS) and optionally rescore the saved val and test predictions of member experiments into a new run with its own row. ms/image is the sum of the members plus the post-processing time.
- **I1**: RT-DETR keeps up to 500 (query, class) pairs (`detections_per_img` now at most `num_queries` times 8 and 500).
- **I2**: `[inference] hflip` predicts flipped frames and mirrors the boxes back, in all four families; the flip TTA is the fusion of the plain and the flipped run.
- **D1**: `[augment]` in the shared loop: flip, photometric distortion, per-batch multi-scale (`train_scales`, `max_size` scaled alike), and `clean_epochs` at the end, after the RT-DETRv2 recipe. All off in the baselines, so existing runs reproduce.
- **D3**: `[ignored_regions] mask` fills ignored regions of training frames with the train mean colour [118, 124, 120], except labelled boxes.
- **I6**: `src/inference/temporal.py` links same-class detections one to one over consecutive frames (greedy by IoU) and moves each score toward the mean or maximum score of its track.

### 10.2 Measured on saved val predictions (laptop, shared evaluator)

Settings were chosen on full val, the frames they are reported on, so the numbers are slightly optimistic. To check selection, every setting is also scored on two disjoint folds of val clips (A: `uav0000086`, `uav0000137`, `uav0000263`, `uav0000339`; B: `uav0000117`, `uav0000182`, `uav0000305`). Rows with a run on the PC replace these numbers once logged.

| Post-processing | Members | mAP | AP50 | AP_small | mAP ≤ 1920 px | fold A | fold B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none (reference) | R | 0.3209 | 0.6027 | 0.1334 | 0.2935 | 0.3386 | 0.2288 |
| temporal, max, weight 0.5, link IoU 0.3 | R | 0.3276 | 0.6124 | 0.1455 | 0.3007 | 0.3422 | 0.2371 |
| NMS IoU 0.7, equal weights | R, F | 0.3196 | 0.6429 | 0.1317 | 0.3018 | 0.3388 | 0.2336 |
| WBF IoU 0.65, equal weights | R, F | 0.3360 | 0.6463 | 0.1382 | 0.3107 | 0.3529 | 0.2473 |
| WBF IoU 0.65, equal weights | R, F, Y | 0.3533 | 0.6702 | 0.1518 | 0.3307 | 0.3639 | 0.2725 |
| WBF as above, then temporal (max, 0.5, link IoU 0.5) | R, F, Y | 0.3617 | 0.6795 | 0.1616 | 0.3408 | 0.3709 | 0.2797 |
| WBF IoU 0.7, weights 0.5 per RT-DETR run, 1 per CNN | 4 × R runs, F, Y | 0.3645 | 0.6798 | 0.1607 | 0.3400 | 0.3743 | 0.2891 |

R `rtdetr-baseline`, F `faster_rcnn-giou_loss`, Y `yolo-baseline`; the four R runs are `baseline`, `lr_scaled`, `lr_scaled_repeat-s7`, `denoising_groups`.

Per-class AP (full val):

| | person | bicycle | car | van | truck | tricycle | bus | motor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R | 0.340 | 0.274 | 0.529 | 0.301 | 0.245 | 0.275 | 0.443 | 0.161 |
| R + temporal | 0.340 | 0.285 | 0.528 | 0.310 | 0.257 | 0.287 | 0.453 | 0.161 |
| WBF R, F | 0.306 | 0.316 | 0.520 | 0.297 | 0.337 | 0.303 | 0.434 | 0.173 |
| WBF R, F, Y | 0.328 | 0.316 | 0.535 | 0.311 | 0.375 | 0.330 | 0.453 | 0.179 |
| WBF R, F, Y + temporal | 0.342 | 0.322 | 0.537 | 0.320 | 0.380 | 0.342 | 0.471 | 0.180 |

Results:

- **WBF against NMS.** At equal weights, WBF of R and F gains 0.0151 mAP and NMS fusion loses 0.0013; WBF IoU 0.6 to 0.7 is a plateau (within 0.0002). Weighting RT-DETR higher (2:1, 3:1) lowers mAP at every IoU.
- **Three families.** WBF of R, F, Y gains 0.0324 mAP, 0.0372 on ≤ 1920 px frames, and gains on both folds (+0.025, +0.044). The six classes other than bus and truck gain 0.020 on average; truck gains the most (+0.130), person loses 0.012. With other RT-DETR members the gain repeats: `lr_scaled` + F + Y 0.3597 (+0.0292 over `lr_scaled`), `lr_scaled_repeat-s7` + F + Y 0.3539 (+0.0328). Two families (R, F) gain less (+0.0151), mostly from truck, bicycle, and tricycle, with person −0.034.
- **Temporal rescoring.** Max beats mean at every setting; weight 0.5 is best. The chosen setting gains 0.0067 on R and holds on the other models without retuning: +0.0045 on `lr_scaled_repeat-s7`, +0.0068 on F, +0.0053 on Y. On top of the three-family WBF it adds 0.0084, on both folds, and every class is then above the reference. AP_small gains the most (+0.012 on R, +0.010 on the fused predictions).
- **More RT-DETR members.** The four RT-DETR runs fused alone give 0.3297 (+0.0088); with F and Y at half weight per RT-DETR run 0.3645, 0.0112 above the three-family fusion. This comes from a six-setting grid and is a candidate, not a result.

Hypotheses (not tested): the ensemble gain comes from complementary errors (RT-DETR confuses truck with van and bus, where Faster R-CNN is stronger); the person loss of the fusion comes from the weaker person AP of F and Y (0.306, 0.312 against 0.340). Temporal rescoring helps because true objects persist over frames while many false positives do not.

### 10.3 GPU queue (phase B)

Ordered by expected value per GPU hour; `scripts/run_queue.sh` runs the items and commits `results.csv` after each. Post-processing items are CPU-only and come first.

| # | Item | Hypothesis | Est. time | Drop if |
| --- | --- | --- | --- | --- |
| 1 | `postprocess/baseline` | reproduces `rtdetr-baseline` (pipeline check) | 2 min CPU | mAP differs from 0.3209 |
| 2 | `postprocess/ens_rtdetr_frcnn_yolo_temporal` | M2 + I6, laptop 0.3617 | 3 min CPU | |
| 3 | `postprocess/ens_rtdetr_frcnn_yolo` | M2 three families, laptop 0.3533 | 3 min CPU | |
| 4 | `postprocess/ens_all_runs` | more RT-DETR members, laptop 0.3645 | 4 min CPU | |
| 5 | `postprocess/ens_rtdetr_frcnn` | M2 two families, laptop 0.3360 | 2 min CPU | |
| 6 | `postprocess/temporal_rtdetr` | I6 on the reference, laptop 0.3276 | 2 min CPU | |
| 7 | `rtdetr/infer_det500` | I1: lower-ranked pairs add recall | 8 min | below +0.002 |
| 8 | `rtdetr/infer_hflip`, then `postprocess/ens_rtdetr_hflip` | I2 flip TTA | 10 min | TTA below +0.003 |
| 9 | `rtdetr/infer_size1280` | I3: small objects gain from upscaling | 12 min | AP_small does not rise |
| 10 | `rtdetr/infer_size1152`, `rtdetr/infer_size1408` | I3 trend around 1280 | 25 min | |
| 11 | `rtdetr/augment_recipe` | D1: augmentation fixes the early peak | 1.6 h | below +0.0094 and not across classes |
| 12 | `rtdetr/mask_ignored` | D3: ignored regions as background hurt | 1.6 h | below +0.0094 and not across classes |
| 13 | `rtdetr/augment_flip_photometric` | D1 ablation without multi-scale | 1.6 h | only if 11 helps |

Total about 1 hour of inference and post-processing plus 4.8 hours of training (each training item includes its val and test prediction).

### 10.4 Proposed additions (need approval before GPU time)

1. **Flip TTA for the ensemble members** (`faster_rcnn/infer_hflip`, `yolo/infer_hflip`): if I2 helps RT-DETR, the flipped CNN predictions are extra fusion members. About 15 min. Drop if I2 on RT-DETR is below +0.003.
2. **Faster R-CNN with the winning D1 or D3 setting** (5 epochs, like `giou_loss`): a stronger second member. About 0.8 h. Only if D1 or D3 helps RT-DETR beyond noise.
3. **M1**: final RT-DETR at step 2 with the winning data setting and `lr_scaled` (milder decline), plus the same for the second-best data setting as a seed check. About 3.5 h each; settings wait for items 11 and 12 and `rtdetr-step2`.
4. **`yolo11m`** as a stronger third member: timing run 10 min, then about 1.5 h. Drop if the timing run needs batch 1.
5. **M7 ignore-aware loss for Faster R-CNN**: about 60 lines and 0.8 h; only if D3 helps, since both test the same hypothesis.

Not proposed: D-FINE / DEIM (new wrapper and timing before Monday), Soft-NMS (240 duplicates at the best-F1 threshold for Faster R-CNN), NWD (loss and matcher changes).

## 11. Pipeline status (2026-11-09, after the first queue)

### 11.1 Measured on the PC (rows in `experiments/results.csv`)

| Experiment | Change | mAP | AP_small | mAP ≤ 1920 px |
| --- | --- | --- | --- | --- |
| `rtdetr-baseline` | reference | 0.3209 | 0.1334 | 0.2935 |
| `rtdetr-augment_recipe` | D1 recipe augmentation | 0.3373 | 0.1449 | 0.3174 |
| `rtdetr-step2` | D2 step 2 frames | 0.3278 | 0.1360 | 0.3045 |
| `rtdetr-infer_det500` | I1 500 detections | 0.3230 | 0.1340 | 0.2954 |
| `rtdetr-infer_size1152` | I3 inference at 1152 px | 0.3254 | 0.1475 | 0.2957 |
| `rtdetr-infer_size1280` | I3 inference at 1280 px | 0.3185 | 0.1500 | 0.2906 |
| `rtdetr-infer_size1408` | I3 inference at 1408 px | 0.3062 | 0.1580 | 0.2810 |
| `rtdetr-infer_hflip` | I2 flipped frames only | 0.3049 | 0.1179 | 0.2737 |
| `postprocess-ens_rtdetr_hflip` | I2 flip TTA, WBF of both | 0.3049 | 0.1151 | 0.2670 |

- D1 is the only change above the 0.0094 seed noise (+0.0164). Its monitoring curve peaks at epoch 2 of 7, as the baseline, but stays near 0.33 afterwards instead of falling to 0.296.
- Larger inference sizes raise AP_small steadily and lower mAP above 1152 px.
- Flip TTA fails because of a systematic offset, not because of the mirroring: on flipped frames `rtdetr-baseline` predicts box centres 2.3 px further right than on the original frames (matched pairs, score ≥ 0.3), the same in every quarter of the frame and 2.0 to 2.4 px at the model input for every frame size. Its AP50 is unchanged (0.6019 against 0.6027) while the higher IoU thresholds lose. Hypothesis: a model trained without flips learns a one-sided correction of its stride asymmetry; `rtdetr-infer_recipe_hflip` (trained with flips) tests this.

### 11.2 Measured on saved val predictions (laptop, shared evaluator)

| Setting | mAP | Fold A | Fold B |
| --- | --- | --- | --- |
| `rtdetr-baseline` + temporal rescoring | 0.3276 | 0.3422 | 0.2371 |
| + box smoothing along tracks | 0.3301 | 0.3444 | 0.2395 |
| + gap filling instead of smoothing | 0.3281 | 0.3426 | 0.2375 |
| R + F + Y, WBF, temporal | 0.3617 | 0.3709 | 0.2797 |
| + box smoothing | 0.3649 | 0.3739 | 0.2828 |
| `augment_recipe` + F + Y, WBF, temporal, smoothing | 0.3724 | 0.3759 | 0.3086 |
| + `yolo-imgsz_1280` | 0.3755 | 0.3769 | 0.3140 |

- Box smoothing replaces every box linked in both neighbouring frames by `(previous + 2 * current + next) / 4`; the gain is mostly AP75 (+0.0041 on `rtdetr-baseline`).
- Extra members on R + F + Y (0.3533): `yolo-imgsz_1280` +0.0063, `yolo-tal_topk` +0.0042, `faster_rcnn-lr_cosine` +0.0028, `retinanet-baseline` +0.0023. The YOLO trained at a second scale adds most.
- Ruled out: higher member NMS IoU (only 1.4% of train and 1.8% of val GT boxes have a same-class neighbour at IoU ≥ 0.5), checkpoint averaging across RT-DETR runs (the WBF of four RT-DETR runs gains only +0.009 over one).

### 11.3 Second queue (2026-11-09 night)

| # | Item | Hypothesis | Est. time |
| --- | --- | --- | --- |
| 1 to 4 | `postprocess/temporal_smooth_rtdetr`, `ens_rtdetr_frcnn_yolo_temporal_smooth`, `ens_recipe_frcnn_yolo_smooth`, `ens_recipe_frcnn_yolo_yolo1280_smooth` | laptop 0.3301, 0.3649, 0.3724, 0.3755 | 15 min CPU |
| 5 | `yolo/infer_imgsz1280` | the second scale alone gives the `yolo-imgsz_1280` member gain | 5 min |
| 6 | `yolo/infer_hflip` | flip TTA works for a model trained with flips | 5 min |
| 7 | `rtdetr/infer_recipe_hflip` | flip training removes the 2.3 px offset | 8 min |
| 8 | `rtdetr/infer_recipe_size1152` | multi-scale training moves the best inference size up | 10 min |
| 9 | `rtdetr/recipe_step2` | M1: D1 and D2 combined, 3 epochs at step 2 | 2.9 h |
| 10, 11 | `rtdetr/infer_recipe_step2_det500`, `infer_recipe_step2_size1152` | I1 and I3 on M1 | 20 min |
| 12 | `rtdetr/ema` | weight EMA against the epoch-to-epoch swings and the decline | 1.6 h |
| 13 | `rtdetr/recipe_step2_repeat` | M1 gain holds with seed 7; second M1 member for the fusion | 2.9 h |

EMA uses decay 0.999 and a 100-step ramp instead of the official 0.9999 and 2,000 steps: with the official values the decay stays below 0.71 until epoch 2, where every run peaks, so the average would equal the trained weights there.

## 12. Pipeline status (2026-11-09, after the second queue)

### 12.1 Measured on the PC (rows in `experiments/results.csv`)

| Experiment | mAP | AP_small | mAP ≤ 1920 px | Best epoch | Reading |
| --- | --- | --- | --- | --- | --- |
| `rtdetr-augment_flip_photometric` | 0.3283 | 0.1259 | 0.3066 | 4 | between 0.3279 and 0.3467: flips and photometric alone +0.0074 over the baseline (within seed noise), so multi-scale carries part of the recipe gain; M1 keeps the full recipe |
| `rtdetr-recipe_step2` (M1) | 0.3402 | 0.1396 | 0.3177 | 1 | +0.0029 over `augment_recipe`, within seed noise |
| `rtdetr-recipe_step2_repeat-s7` | 0.3404 | 0.1356 | 0.3127 | 1 | M1 repeats with seed 7 |
| `rtdetr-infer_recipe_step2_det500` | 0.3417 | 0.1403 | 0.3191 | | best single model on val |
| `rtdetr-infer_recipe_step2_size1152` | 0.3392 | 0.1497 | 0.3168 | | AP_small up, mAP down, as on the baseline |
| `rtdetr-infer_recipe_size1152` | 0.3414 | 0.1546 | 0.3212 | | +0.0041 on `augment_recipe` |
| `rtdetr-infer_recipe_hflip` | 0.3382 | 0.1394 | 0.3190 | | flipped frames alone now score as high as unflipped ones (0.3373): flip training removed the offset |
| `rtdetr-ema` | 0.3324 | 0.1336 | 0.3102 | 2 | +0.0115 over the baseline, one seed; monitoring mAP still falls from 0.3334 (epoch 2) to 0.3085 (epoch 7) |
| `yolo-infer_imgsz1280` | 0.2533 | 0.1017 | 0.2352 | | below `yolo-baseline` 0.2671: the `yolo-imgsz_1280` member gain needs the training at 1280 |
| `yolo-infer_hflip` | 0.2621 | 0.1105 | 0.2402 | | |
| `postprocess-ens_recipe_frcnn_yolo_yolo1280_smooth` | 0.3755 | 0.1772 | 0.3559 | | reproduces the laptop grid |

### 12.2 Final ensemble, forward selection on saved val predictions (laptop, shared evaluator)

All settings: WBF IoU 0.65, temporal max, weight 0.5, link IoU 0.5, box smoothing. Fold A is `uav0000086`, `uav0000137`, `uav0000263`, `uav0000339`; fold B is `uav0000117`, `uav0000182`, `uav0000305`.

| Members | mAP | Fold A | Fold B |
| --- | --- | --- | --- |
| B = `augment_recipe`, `faster_rcnn-giou_loss`, `yolo-baseline`, `yolo-imgsz_1280` | 0.3755 | 0.3769 | 0.3140 |
| B + `infer_recipe_step2_det500` | 0.3813 | 0.3821 | 0.3189 |
| B + `infer_recipe_step2_det500`, `recipe_step2_repeat-s7` | 0.3824 | 0.3853 | 0.3179 |
| C = B + those two + `infer_recipe_hflip` | 0.3829 | 0.3864 | 0.3208 |
| D = C + `faster_rcnn-lr_cosine` (`ens_rtdetr4_frcnn2_yolo2_smooth`) | 0.3859 | 0.3882 | 0.3215 |

On D, adding `faster_rcnn-anchors_small`, `yolo-scale_025`, `retinanet-baseline`, `faster_rcnn-baseline`, `faster_rcnn-baseline_5ep-s7`, or `yolo-tal_topk`, weights 1.5 for the other families, WBF IoU 0.6, or temporal weight 0.6 changed mAP by −0.0021 to +0.0006. Per-class AP of D: person 0.356, bicycle 0.325, car 0.555, van 0.344, truck 0.417, tricycle 0.363, bus 0.548, motor 0.179.

### 12.3 Submissions

Chosen on val before any LB score: the final two Kaggle selections are D (best val) and B (members fixed before the second queue, the hedge against selection on 7 val clips). Two further submissions give test checks for the report: the best single model `rtdetr-infer_recipe_step2_det500`, and `postprocess-temporal_smooth_rtdetr`, a single factor over `rtdetr-baseline` (LB 0.2563).

### 12.4 Val and LB gap (2026-11-09, laptop, saved predictions)

Checked and found correct: test JPEG sizes equal `test_images.csv`; test frames are consecutive in every clip; test goes through the same `predict_frames` as val; predicted boxes scale with the frame size; scoring val without ignored regions or with 100 detections per frame lowers every model by 0.007 to 0.03 and does not reproduce the RetinaNet LB drop.

Frame sizes: train and val have no frames below 1344 px; test has 321 frames at 680×382 and 279 at 960×540 (16%, apparently 2× downscaled), 1,915 at 1344 or 1360 px, and none above 1920 px. Val has 582 frames (22%) at 2688 or 2720 px, with median object √area 65 to 73 px.

Val mAP by frame width against public LB:

| Model | Val 1344 px | Val 1904 px | Val 2688+ px | Full val | LB |
| --- | --- | --- | --- | --- | --- |
| `yolo-baseline` | 0.269 | 0.233 | 0.317 | 0.267 | 0.210 |
| `faster_rcnn-baseline` | 0.264 | 0.281 | 0.329 | 0.301 | 0.227 |
| `retinanet-baseline` | 0.230 | 0.276 | 0.281 | 0.269 | 0.156 |
| `rtdetr-baseline` | 0.281 | 0.289 | 0.332 | 0.321 | 0.256 |
| `ens_recipe_frcnn_yolo_yolo1280_smooth` | 0.349 | 0.339 | 0.404 | 0.375 | |
| `ens_rtdetr4_frcnn2_yolo2_smooth` | 0.354 | 0.342 | 0.406 | 0.386 | |

Per-clip val mAP of `rtdetr-baseline` follows the median object size (0.215 at 32 px for `uav0000182`, 0.376 at 73 px for `uav0000137`); `uav0000263`, moved into val by the swap, scores 0.253. Larger inference sizes do not raise mAP on the 1344 px val frames (1024: 0.281, 1152: 0.278, 1280: 0.275, 1408: 0.271).

Hypothesis, not tested: the gap comes from smaller objects on test (half-resolution frames, no frames above 1920 px), not from the pipeline.

LB of the four submissions (`results.csv`): `ens_rtdetr4_frcnn2_yolo2_smooth` 0.2981, `ens_recipe_frcnn_yolo_yolo1280_smooth` 0.2877, `rtdetr-infer_recipe_step2_det500` 0.2722, `postprocess-temporal_smooth_rtdetr` 0.2688 (+0.0125 over `rtdetr-baseline`, val +0.0092). The LB order equals the val order.

All nine public LB scores (Kaggle submission list, 2026-11-09):

| Submission | Val mAP | LB | Val minus LB |
| --- | --- | --- | --- |
| `ens_rtdetr4_frcnn2_yolo2_smooth` (D) | 0.3859 | 0.2981 | 0.0878 |
| `ens_recipe_frcnn_yolo_yolo1280_smooth` (B) | 0.3755 | 0.2877 | 0.0878 |
| `rtdetr-infer_recipe_step2_det500` | 0.3417 | 0.2722 | 0.0695 |
| `rtdetr-lr_scaled` | 0.3305 | 0.2431 | 0.0874 |
| `postprocess-temporal_smooth_rtdetr` | 0.3301 | 0.2688 | 0.0613 |
| `rtdetr-baseline` | 0.3209 | 0.2563 | 0.0646 |
| `faster_rcnn-baseline` | 0.3013 | 0.2273 | 0.0740 |
| `retinanet-baseline` | 0.2689 | 0.1562 | 0.1127 |
| `yolo-baseline` | 0.2671 | 0.2097 | 0.0574 |

Spearman rank correlation between val and LB is 0.93 (Pearson 0.92, 9 submissions). The order differs in two places, and both pairs are within the 0.0094 seed spread on val:
- `rtdetr-lr_scaled`: +0.0096 over `rtdetr-baseline` on val, −0.0132 on LB. Its seed 7 repeat (0.3211) had already shown that the val gain was noise.
- `retinanet-baseline` against `yolo-baseline`: 0.0018 apart on val.

Ensembling keeps about 60% of its val gain on the LB: D over `infer_recipe_step2_det500` is +0.0442 on val and +0.0259 on LB.

### 12.5 Low-resolution augmentation (decided 2026-11-09)

Deviation from data rule 4, decided by the user: the experiment is motivated by test frame sizes (600 frames at 680×382 and 960×540, §12.4), not by train or val. It is measured on val only.

| # | Item | Hypothesis | Cost | Keep if |
| --- | --- | --- | --- | --- |
| 1 | `rtdetr/infer_recipe_step2_half` | M1 loses accuracy on frames shrunk 2× (val stand-in for the half-resolution test frames) | 10 min | reference for 3 |
| 2 | `rtdetr/recipe_step2_low_res` | M1 with a quarter of the augmented frames shrunk 2× (`low_res_prob` 0.25, `low_res_factor` 0.5) is robust to half-resolution frames | about 1.7 h | full val mAP at least 0.3308 (0.3402 minus the 0.0094 seed noise) |
| 3 | `rtdetr/infer_recipe_step2_low_res_half` | as 2, on half-resolution val | 10 min | above 1 by more than 0.0094 |
| 4 | `rtdetr/infer_recipe_step2_low_res_det500` | ensemble member replacing `infer_recipe_step2_det500` | 10 min | the ensemble with it scores higher on val |

Shrunk 2×, the val widths 1344, 1904, 2688, and 2720 px become 672, 952, 1344, and 1360 px, all close to test widths; the shrunk 1344 and 1904 px clips stand in for the half-resolution test frames. Antialiased shrinking of a sharp JPEG is a stand-in, not a copy, of how the test frames were made. The clean last epoch of M1 trains without the low-resolution draw.

Results (items 1 to 3: rows in `results.csv`; item 4: saved val predictions, laptop, settings of §12.2):

| # | Run | mAP | AP_small | Keep if | Outcome |
| --- | --- | --- | --- | --- | --- |
| 1 | `infer_recipe_step2_half` | 0.3238 | 0.0957 | reference | M1 loses 0.0164 on half-resolution val |
| 2 | `recipe_step2_low_res` | 0.3311 | 0.1404 | at least 0.3308 | passes |
| 3 | `infer_recipe_step2_low_res_half` | 0.3296 | 0.1196 | above 0.3332 | fails (+0.0058) |
| 4 | D with `infer_recipe_step2_low_res_det500` instead of `infer_recipe_step2_det500` | 0.3842 (A 0.3881, B 0.3204) | | above 0.3859 | fails |
| 4 | D plus `infer_recipe_step2_low_res_det500` | 0.3843 (A 0.3890, B 0.3164) | | above 0.3859 | fails |

mAP by val frame width:

| Run | 1344 px | 1904 px | 2688+ px | AP_small 1344 px |
| --- | --- | --- | --- | --- |
| `recipe_step2` | 0.2975 | 0.3088 | 0.3551 | 0.1955 |
| `infer_recipe_step2_half` | 0.2489 | 0.3071 | 0.3498 | 0.1242 |
| `recipe_step2_low_res` | 0.2859 | 0.3013 | 0.3568 | 0.1920 |
| `infer_recipe_step2_low_res_half` | 0.2679 | 0.3054 | 0.3554 | 0.1579 |

Dropped, with no train and val version. The low-resolution draw trades mAP on the full-resolution 1344 px clips (−0.0116) for mAP on the same clips shrunk to 672 px (+0.0190), and test has about three times more 1344 or 1360 px frames than half-resolution ones.

Estimate, not measured on test: shrinking to 672 px costs M1 0.0486 on the 1344 px clips. Weighted by the 16% half-resolution test frames, that is below 0.01 of the 0.07 gap between val and LB (0.3417 against 0.2722). The half-resolution frames therefore explain only a small part of the gap, which weakens the §12.4 hypothesis. M1 on the 1344 px val clips (0.2975) is closer to its LB score than full val is.

### 12.6 Train and val models (built 2026-11-09)

`train_splits = ["train", "val"]` trains the shared-loop families on all 63 clips at the `train_frames` step. No frames are held out, so the per-epoch monitoring is skipped, `best.pt` is the last epoch, and the val metrics of these runs are not comparable with any other row. Each config stops at the best epoch of its val run under a constant LR, so it trains with the same LR as that epoch:

| Config | Val run and best epoch | Epochs | Ensemble member it replaces |
| --- | --- | --- | --- |
| `rtdetr/trainval_recipe_step2` (+ `infer_trainval_recipe_step2_det500`) | `recipe_step2`, 1 of 3 | 1 at step 2 | `infer_recipe_step2_det500` |
| `rtdetr/trainval_recipe_step2_repeat` (seed 7) | `recipe_step2_repeat-s7`, 1 of 3 | 1 at step 2 | `recipe_step2_repeat-s7` |
| `rtdetr/trainval_recipe_step5` (+ `infer_trainval_recipe_step5_hflip`) | `augment_recipe`, 2 of 7 | 2 at step 5 | `augment_recipe`, `infer_recipe_hflip` |
| `faster_rcnn/trainval_giou_loss` | `giou_loss`, 2 of 5 | 2 at step 5 | `giou_loss` |
| `faster_rcnn/trainval_baseline_2ep` | `baseline`, 2 of 10 | 2 at step 5 | `lr_cosine` (best epoch 2 of 10, cosine LR 0.90 to 1.0 times the constant one) |

YOLO stays train only: its best epochs (7 and 2 of 20) fall under the decaying Ultralytics LR and `close_mosaic`, which a shorter run does not reproduce. The single-model public LB check is `infer_trainval_recipe_step2_det500` against `infer_recipe_step2_det500` (0.2722), one factor (the training clips). Step 2 for 1 epoch and step 5 for 2 epochs are within seed noise on val (0.3402 and 0.3373) and both are ensemble members, so the ensemble needs no choice between them.

Results (rows in `results.csv`): every run trained like its reference. The train loss per epoch matches it within 0.4:

| Run | Train loss | Reference |
| --- | --- | --- |
| `rtdetr-trainval_recipe_step2` | 17.94 | 18.19 |
| `rtdetr-trainval_recipe_step2_repeat-s7` | 17.76 | 17.95 |
| `rtdetr-trainval_recipe_step5` | 22.40, 15.23 | 22.74, 15.45 |
| `faster_rcnn-trainval_giou_loss` | 0.9962, 0.8196 | 0.9974, 0.8182 |
| `faster_rcnn-trainval_baseline_2ep` | 0.8567, 0.6804 | 0.8590, 0.6749 |

Their val mAP (0.4417 to 0.4934) is measured on training clips and is not comparable with any other row. Public LB: not submitted yet.

### 12.7 Next steps (proposal, 2026-11-09)

1. LB check of the train and val models: `rtdetr-infer_trainval_recipe_step2_det500` against `infer_recipe_step2_det500` (0.2722), and `postprocess-ens_trainval_rtdetr4_frcnn2_yolo2_smooth` (D′) against D (0.2981).
2. Final two Kaggle selections (TODO 5.7), fixed before the D′ score: D′ and D. D′ is the validated recipe of D trained on 63 clips instead of 56, but it cannot be checked on val. D is the best validated pipeline and covers the case where training on val hurts. B leaves the selection, because D scores higher on both val folds. Only if D′ scores below B (0.2877) on the LB, treat it as a defect and select D and B.
3. Cross-validation (TODO 5.4 and 5.5): the final RT-DETR recipe (1 epoch at step 2) and the Faster R-CNN GIoU member (2 epochs at step 5) on 3 `GroupKFold` folds over all 63 clips. Estimated from the train and val runs at about 0.5 h and 0.6 h per fold, so about 3.5 h of GPU time for both. A two-family ensemble can then be scored on each fold on the laptop.
4. `06_final_pipeline.ipynb` (TODO 5.6) on the laptop while the folds run.

LB of the train and val models (2026-11-09): `rtdetr-infer_trainval_recipe_step2_det500` 0.2602 (−0.0120 against 0.2722), D′ 0.2884 (−0.0097 against D 0.2981, +0.0007 against B). Training on val lowered both LB scores. Under the rule in item 2, which was fixed before these scores, D′ stays above B and the selection stays D and D′. Hypotheses, not tested: one seed per model (val seed spread 0.0094); the val clips add frames at 1904 and 2688 px, while test has none above 1920 px; one epoch on 12% more frames means more steps at the constant LR than the best epoch of the val run.

Not proposed: YOLO train and val versions (no faithful shorter schedule, §12.6); a third RT-DETR seed (+0.0011 for the second on val, within noise).

### 12.8 Cross-validation (built 2026-11-09, TODO 5.4)

Folds: `[folds]` in `config/data.toml`, a shuffled `GroupKFold` by clip over all 63 clips with seed 42 (`assign_clip_folds`), 21 clips each:

| Fold | Val frames | Val clips by width (1344 / 1360 / 1904 / 1920 / 2688 / 2720 / 3840 px) | Clips of the adopted val split |
| --- | --- | --- | --- |
| `fold_0` | 9,884 | 5 / 0 / 11 / 0 / 2 / 2 / 1 | 3 |
| `fold_1` | 9,423 | 3 / 1 / 12 / 0 / 5 / 0 / 0 | 3 |
| `fold_2` | 7,740 | 5 / 0 / 9 / 1 / 5 / 1 / 0 | 1 |

The unshuffled `GroupKFold` would balance frames better (about 9,000 per fold), but AGENTS.md fixes seed 42, which only the shuffled variant uses.

`clip_split` (every baseline, `main` by default) selects the adopted split or `fold_<k>`: val is that fold, train the other two folds (42 clips), so per-epoch monitoring, `src.predict`, `src.evaluate`, and post-processing run on the fold unchanged. Runs created before the field read as `main`. YOLO supports only `main`. A post-processing run requires all of its members to share its `clip_split`.

| Config (k = 0, 1, 2) | Recipe | Estimated PC time per fold |
| --- | --- | --- |
| `rtdetr/cv_recipe_step2_fold<k>` | `trainval_recipe_step2` (1 epoch at step 2) with 500 detections, as `infer_recipe_step2_det500` | about 0.6 h (training 0.4 h, val and test prediction 0.2 h) |
| `faster_rcnn/cv_giou_loss_fold<k>` | `trainval_giou_loss` (2 epochs at step 5) | about 0.75 h |
| `postprocess/cv_ens_rtdetr_frcnn_smooth_fold<k>` | WBF of the two with the settings of D | minutes |

Fold mAPs are measured on other clips than the adopted val split and are compared only with each other (mean and spread over the 3 folds, RT-DETR alone against the two-family ensemble), not with other rows.

### 12.9 Misses, stacking, and the Faster R-CNN recipe (2026-11-09)

Misses of D on val (laptop, saved predictions; same-class box at IoU 0.5 or above):

| Size | Objects | Found at any score | Found at score 0.3 or above | No box with IoU 0.1 or above | Same-class box at IoU 0.1 to 0.5 |
| --- | --- | --- | --- | --- | --- |
| small | 20,546 | 0.805 | 0.630 | 0.042 | 0.141 |
| medium | 72,383 | 0.929 | 0.814 | 0.010 | 0.053 |
| large | 12,311 | 0.983 | 0.923 | 0.004 | 0.011 |

Of the 21,987 objects without a box at score 0.3 or above, 7.5% have no box near them at all, 77% are occluded, and 35% are small. A second detector trained on frames with the detected objects masked out (proposed and not built) could recover at most that 7.5%; most misses have a box that is ranked too low or placed too far off.

Stacking (laptop, prototype outside the repository): gradient boosting (scikit-learn `HistGradientBoosting`) rescoring every fused box of D from its score, class, size, the matched score and IoU of each member, the number of agreeing members, and its temporal track. Target: matched at IoU 0.5 by greedy class-aware matching in score order, as COCO AP; boxes in ignored regions left out of training. Leave-one-clip-out over the 7 val clips:

| Scores | mAP | Fold A | Fold B |
| --- | --- | --- | --- |
| D | 0.3859 | 0.3882 | 0.3215 |
| classifier, all features | 0.3577 | 0.3739 | 0.2783 |
| geometric mean of classifier and D score | 0.3720 | 0.3850 | 0.2892 |
| IoU regressor | 0.3501 | 0.3627 | 0.2738 |
| geometric mean of regressor and D score | 0.3673 | 0.3792 | 0.2860 |
| classifier, box features only, regularized | 0.3716 | 0.3857 | 0.2847 |
| geometric mean of that classifier and D score | 0.3795 | 0.3888 | 0.2963 |

Dropped: no variant reaches D, and fold B (with `uav0000182`, the clip with the smallest objects) loses 0.025 to 0.043 in every variant. Hypothesis, not tested: 6 training clips are too few for the rescoring to transfer to an unseen clip; the hand-set WBF and temporal scores carry no fitted parameters to transfer.

`faster_rcnn/giou_recipe` (built, to run after the folds): `giou_loss` with the augmentation of `rtdetr-augment_recipe` (flip, photometric, multi-scale 832 to 1216 px, clean last epoch), 4 epochs, batch 2 with gradient accumulation 2 (effective 4, frozen BatchNorm) because 1216 px at batch 4 would reach about 12 GB. Keep if its val mAP beats `giou_loss` (0.3066) by more than 0.0094 and swapping it into D raises both val folds; about 1 GPU hour.

