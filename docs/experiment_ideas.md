# Experiment ideas

This is a plan, not a list of tasks. Implement an experiment only when asked. Each experiment changes one factor against its family baseline (AGENTS.md, Data and Experiments). Numbers in brackets refer to sections of `notebooks/01_eda.ipynb`.

"Required" items are copied from the assignment rubric; everything else is optional.

## Pipeline (all families)

Already fixed in AGENTS.md: up to 500 detections per frame, per-class NMS with IoU ≥ 0.5, `maxDets` 500, full val as primary metric, frame step 5 for tuning and 2 for final models, the adopted split.

Ideas:
- **Ignored-region masking**: fill ignored regions with the exact train mean colour, keeping pixels inside labelled boxes. 0.60% of objects lie fully inside ignored regions, 1.07% are more than half covered [§7.3]. Compute the mean colour over all train frames when building it.
- **Brightness or gamma augmentation**: 13.2% of sampled val frames are dark (mean luminance < 0.25) versus 2.1% of train [§12.3].
- **Generic scale augmentation** (random resize). No data-based claim; test sizes are not a justification.
- **Frame sampling**: step 2 versus step 1 at the same iteration budget; random start offset within each step window per epoch, seeded [§9.1].
- **Class imbalance** (one change at a time; never combine repeat factor sampling and class-weighted loss in one experiment):
    - Repeat factor sampling (LVIS, Gupta et al. 2019). Compute class frequency per clip as well as per frame: rare classes appear in few clips, so per-frame repetition mostly repeats the same scenes.
    - Class-weighted classification loss, weights ∝ (1 / n_c)^β with β in [0.3, 1], normalised to mean 1, or effective-number weights (Cui et al. 2019).
    - Decoupled training (Kang et al. 2020): fine-tune normally, then train only the classification branch with class-balanced sampling, lower LR, few epochs.
    - Copy-paste of rare-class objects. Ultralytics `copy_paste` needs segmentation masks, so a box-only version would have to be written.
    - Start with per-class AP and a confusion matrix with a background row and column: if most errors are target vs background, imbalance methods help little.
- **Inference**: SAHI slicing, Soft-NMS, horizontal-flip TTA, Weighted Boxes Fusion for ensembles.
- **Evaluator notes**: treat per-class AP for bus (2 val clips, 6 tracks) and truck (15 tracks) as noisy [§9.2, §11.3]. Val bicycle is mostly one clip and mostly ridden: 88.6% partially occluded, 25.5% with IoU > 0.5 with another object [§6.3, §8].
- **Failure-case example**: truck box over a car in `uav0000263_03289_v`, frame 242 [§10]. Labels are not edited.

## Two-stage anchor-based (Faster R-CNN ResNet50-FPN v2)

Required:
- Train from pretrained weights under the fixed protocol; document the training setup.
- Investigate at least two parameters relevant to two-stage detection (e.g. number of RPN proposals, RoI sampling ratio, box regression loss, LR schedule).
- Report mAP, AP50, AP_small, and inference speed on validation.
- Explain what the region proposal stage and the second stage each contribute.
- Analyze at least three failure cases with plausible causes.

Ideas:
- **Anchors, step 1** (3 per location, COCO RPN head weights still load): sizes `((16,), (32,), (64,), (128,), (256,))`, aspect ratios `(0.5, 1, 2)`. Best grid IoU < 0.5: 2.0% of boxes (default 10.5%), 3.6% of small boxes (default 37.0%); < 0.7 gets worse, 68.0% vs 56.8% [§13.3].
- **Anchors, step 2** (6 per location; RPN head output shape changes, so its COCO weights cannot load, which is a second change to report): sizes `((16, 24), (32, 48), (64, 96), (128, 192), (256, 384))`. Below 0.5: 0.8% of boxes, 2.8% of small; below 0.7: 28.5%. Review RPN pre- and post-NMS top-n in the same experiment (about 700k anchors per image at P2).
- RPN proposal counts, RoI sampling ratio, GIoU box loss, cosine schedule with warmup.
- Not worth it: aspect ratio 1.5 (at most 0.4 points at IoU 0.5).

## One-stage anchor-based (RetinaNet ResNet50-FPN v2)

Required:
- Train under the same protocol.
- Tune relevant parameters (e.g. anchor scales and aspect ratios, focal loss parameters, IoU thresholds for positive and negative assignment). Ground the anchor configuration in the object sizes from the data analysis.
- Discuss whether removing the proposal stage can cause issues and how they can be addressed.
- Compare the initial and tuned models with the tuned two-stage detector.

Ideas:
- **Anchors**: base sizes 16 to 256, three scales per octave, aspect ratios `(0.5, 1, 2)`; still 9 per location, so the head loads COCO weights. Below 0.5: 1.6% of boxes (default 10.3%), 5.2% of small (default 37.5%), 0.5% of large [§13.3].
- Focal loss α and γ; positive and negative IoU thresholds (defaults 0.5 and 0.4).

## One-stage anchor-free (YOLO11)

Required:
- Train under the same protocol.
- Tune at least two relevant parameters (e.g. box regression loss, label assignment strategy, input resolution, an additional high-resolution detection head).
- Explain how locations are assigned to objects without anchors and how this affects small and densely packed objects.
- Report the search range, the selected settings, and validation results.
- Compare the initial and tuned models with the tuned anchor-based detectors, with results broken down by object size.

Ideas:
- **Input resolution**: `imgsz` 1824 (baseline, same object scale as torchvision) versus 1280. At 1280 the median √area is about 30 px and about half of train objects fall below 32 px [§5].
- **P2 detection head** (`yolo11-p2` model config): extra stride-4 level for small objects; measure speed and VRAM cost.
- Box loss or assigner settings (TAL top-k, loss weights).

## Query-based (RT-DETRv2, Hugging Face)

Required:
- Train under the same compute budget.
- Tune relevant parameters (e.g. number of queries relative to scene density, matching cost weights, auxiliary or denoising losses, separate learning rates for backbone and transformer).
- Explain how bipartite matching removes the need for NMS and how this changes training dynamics.
- Plot validation mAP over training against a CNN-based detector given the same budget; discuss small objects and dense scenes.
- If the model has not converged within the budget, state it and explain what you observe.

Ideas:
- **Number of queries**: the default 300 already exceeds the densest frame (180 objects; no frame above 300) [§6.1, §6.2]. Test 300 versus a larger value only if recall on dense frames is low.
- Matcher cost weights (class / bbox / GIoU); denoising on vs off; separate backbone and transformer learning rates.
- Checkpoint: R18 (faster) vs R50 (more accurate). TODO: choose.

## Your pipeline

Required:
- Push further for the best leaderboard score, building on the required model comparisons through data and inference strategies and additional modeling experiments, with consistent validation.
- Both parts (data or inference strategy, and modeling) are graded separately; include intermediate comparisons that separate their effects.
- A final pipeline justified by earlier results, reproducible code, and a summary comparison of all models.

Ideas: the Pipeline section above, plus an ensemble of the best two or three families with Weighted Boxes Fusion. Check final candidates with the 3-fold `GroupKFold` over all 63 labelled clips.

## Bonus

- **Data temporality (1 pt)**: link detections across consecutive frames by IoU (median IoU of the same object at k = 1 is 0.897 train, 0.868 val) and rescore them along the track, or propagate confident detections. Needs a fallback for fast motion (val 10th percentile IoU 0.50) and must allow class changes along a track (153 train, 8 val tracks change class) [§9]. Test clips are short (median 74 frames), so tracks are short.
- **Visualization (1 pt)**: W&B logging of curves, configs, and validation predictions for every required model; plus at least one of RoI or query embeddings (t-SNE / UMAP), transformer attention maps, Grad-CAM on FPN levels.
- **Custom architectural decisions (up to 3 pts)**: e.g. a P2 level or QueryDet-style high-resolution path, with full experimental evidence.