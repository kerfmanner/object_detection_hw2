# Literature

Reading list from the Kaggle competition page (Practical Tips → Reading materials), copied with the organizers' descriptions. Fill the "Used in" column with where each source is used (notebook section or `src/` module); the report must cite every source that has an entry there.

Public links (papers, docs, repositories) may be opened to check implementation details; cite what you use. The Kaggle competition page needs a login: everything needed from it is in AGENTS.md. Opening a source never permits downloading its data or checkpoints (Hard Rule 6).

Course citation: Oleh Basystyi. Multiple Object Video Detection | HW2 | UCU AI 2026. https://www.kaggle.com/competitions/multiple-object-video-detection-hw-2-ucu-ai-2026, 2026. Kaggle.

## General background

| Resource | Description | Used in |
| --- | --- | --- |
| [Object Detection in 20 Years: A Survey](https://arxiv.org/abs/1905.05055) | This survey offers a broad overview of how detection evolved from hand-crafted features to modern deep detectors. Read it first to see where each model family in this assignment fits. | |
| [Towards Large-Scale Small Object Detection: Survey and Benchmarks](https://arxiv.org/abs/2207.14096) | This survey reviews why small objects are hard and which techniques help. This is relevant for the homework, since many targets in the VisDrone dataset occupy only a few pixels. | |
| [Speed/accuracy trade-offs for modern convolutional object detectors](https://arxiv.org/abs/1611.10012) | This work compares detector families under controlled settings. This will serve as a good model for how to set up fair comparisons and report inference speed. | |
| [Lilian Weng's "Object Detection for Dummies" series](https://lilianweng.github.io/) | This is a walkthrough of object detection, from R-CNN to YOLO, SSD and RetinaNet. This resource will be useful for building intuition before reading the original papers. | |
| [Dive into Deep Learning, Computer Vision chapter](https://d2l.ai/chapter_computer-vision/index.html) | This explains anchor boxes, IoU, NMS and multiscale detection with runnable code. Helpful if you want to understand these components by implementing them. | |
| [Karpathy, "A Recipe for Training Neural Networks"](https://karpathy.github.io/2019/04/25/recipe/) | Here you can find practical advice on debugging and iterating on training pipelines. Follow it to avoid wasting GPU hours on silent bugs. | |
| [VisDrone](https://arxiv.org/abs/2001.06303) | This describes how the data was collected and annotated, including ignored regions. Read this before writing your conversion code. | |

## Data augmentation and training

| Resource | Description | Used in |
| --- | --- | --- |
| [Albumentations docs](https://albumentations.ai/docs/) and [paper](https://arxiv.org/abs/1809.06839) | This is a fast augmentation library that transforms bounding boxes together with images. The easiest way to add reproducible augmentation to a custom dataloader. | |
| [Bag of Freebies for Training Object Detectors](https://arxiv.org/abs/1902.04103) | This write-up many training tricks (mixup, label smoothing, cosine schedules) that improve detectors at no inference cost. You can use this as inspiration for additional experiments on top of your selected baselines. | |
| [YOLOv4](https://arxiv.org/abs/2004.10934) | Introduces Mosaic augmentation and systematically tests many training tricks. Useful both as a catalogue of ideas and as an example of ablation-driven design. | |
| [Simple Copy-Paste](https://arxiv.org/abs/2012.07177) | This work shows that pasting objects between images is a strong augmentation. Worth trying for rare classes. | |
| [Normalized Wasserstein Distance](https://arxiv.org/abs/2110.13389) | Proposes a similarity measure for tiny boxes, where IoU is overly sensitive to small shifts. A natural candidate for the required loss-modification experiment. | |
| [GIoU](https://arxiv.org/abs/1902.09630) and [DIoU/CIoU](https://arxiv.org/abs/1911.08287) | IoU-based box regression losses that give useful gradients even when boxes don't overlap. They are the standard alternatives to L1/Smooth L1 and easy to swap into most frameworks. Use this as inspiration for the loss experiments where appliccable. | |
| [scikit-learn GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) | Splits data so that all frames from one sequence land in the same fold. Use it to build sequence-level splits without leakage. | |

## Evaluation

| Resource | Description | Used in |
| --- | --- | --- |
| [COCO detection metrics](https://cocodataset.org/#detection-eval) and [COCO paper](https://arxiv.org/abs/1405.0312) | Define mAP, AP50 and AP_small exactly as you are asked to report them. Check these definitions so your numbers are comparable across models. This will also help you understand and explain metrics during the defense. | |
| [pycocotools](https://github.com/cocodataset/cocoapi) / [faster-coco-eval](https://github.com/MiXaiLL76/faster_coco_eval) | The reference COCO evaluator and a faster drop-in replacement. Marking ignored regions as iscrowd=1 makes the evaluator skip detections inside them. | |
| [TorchMetrics MeanAveragePrecision](https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html) | A COCO-style mAP implementation that plugs into PyTorch training loops. Convenient for tracking validation mAP during training. | |
| [TIDE](https://arxiv.org/abs/2008.08115) ([code](https://github.com/dbolya/tide)) | Breaks detection errors into classification, localization, duplicate, background and missed-object types. This is very useful for the failure-case analysis, since it tells you which error type costs the most mAP. This can be a strong tool for improving your leaderboard score. | |
| [FiftyOne](https://docs.voxel51.com/) | A tool for browsing predictions against ground truth and filtering by confidence, size or error type. Helps you find failure cases visually instead of scrolling through images. | |

## Inference strategies

| Resource | Description | Used in |
| --- | --- | --- |
| [SAHI](https://arxiv.org/abs/2202.06934) ([code](https://github.com/obss/sahi)) | Runs detection on overlapping image slices and merges the results. A simple way to improve small-object recall on high-resolution frames without retraining. | |
| [Soft-NMS](https://arxiv.org/abs/1704.04503) | Decays the scores of overlapping boxes instead of removing them. Often helps in dense scenes where standard NMS suppresses true neighbours. | |
| [Weighted Boxes Fusion](https://arxiv.org/abs/1910.13302) | Averages overlapping boxes from several models or TTA passes instead of discarding them. The algorithm is short enough to implement yourself from the paper for ensembling. | |

## Bonus: data temporality

| Resource | Description | Used in |
| --- | --- | --- |
| [Seq-NMS](https://arxiv.org/abs/1602.08465) | Links high-scoring boxes across frames and rescores them along the sequence. One of the simplest post-processing methods for exploiting temporal consistency. | |
| [T-CNN](https://arxiv.org/abs/1604.02532) | Propagates detections to neighbouring frames and rescores them using tubelets. Directly matches the "propagate confident detections" idea in the bonus description. | |
| [Detect to Track and Track to Detect](https://arxiv.org/abs/1710.03958) | Trains detection and frame-to-frame tracking jointly. Shows how tracking signals can improve per-frame detection. | |
| [FGFA](https://arxiv.org/abs/1703.10025), [SELSA](https://arxiv.org/abs/1907.06390), [MEGA](https://arxiv.org/abs/2003.00586) | Feature-level temporal aggregation methods for video detection. Read these if you want to go beyond post-processing. | |
| [SORT](https://arxiv.org/abs/1602.00763) and [ByteTrack](https://arxiv.org/abs/2110.06864) | Lightweight tracking-by-detection methods. ByteTrack's use of low-confidence boxes is especially relevant for recovering missed detections. | |
| [TorchVision RAFT optical flow](https://docs.pytorch.org/vision/main/models/raft.html) and [Kornia](https://kornia.readthedocs.io/) | A pretrained optical flow model and a differentiable geometry library with homography estimation. Use them to compensate for camera motion before matching boxes across frames. | |

## Bonus: visualization

| Resource | Description | Used in |
| --- | --- | --- |
| [Grad-CAM](https://arxiv.org/abs/1610.02391) | Explains predictions by highlighting the image regions that drive them. Read it to understand what the heatmaps do and don't show. | |
| [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) | A library with ready examples for Faster R-CNN and YOLO. Saves you from writing detector-specific hooks by hand. | |
| [DETR repository](https://github.com/facebookresearch/detr) | Includes a notebook visualizing decoder attention for individual queries. A good template for the attention-map analysis. | |
| [UMAP](https://arxiv.org/abs/1802.03426) ([docs](https://umap-learn.readthedocs.io/)) | A fast dimensionality reduction method for projecting embeddings to 2D. Useful for plotting RoI features or query embeddings. | |
| [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/) | An interactive article on how t-SNE plots can mislead. Read it before drawing conclusions from cluster shapes. | |
| [W&B media logging](https://docs.wandb.ai/guides/track/log/media) | Explains how to log images with bounding boxes. The free personal and academic tier covers the logging the bonus requires. | |

## Bonus: custom architectural decision

| Resource | Description | Used in |
| --- | --- | --- |
| [QueryDet](https://arxiv.org/abs/2106.09018) | Uses coarse predictions to decide where to run high-resolution detection. An example of an architectural change aimed specifically at small objects. | |
| [ClusDet](https://arxiv.org/abs/1904.08008) | Detects object clusters in aerial images first, then runs fine detection inside them. Relevant for scenes where small objects are unevenly distributed. | |

## Models: two-stage anchor-based

| Resource | Description | Used in |
| --- | --- | --- |
| [Faster R-CNN](https://arxiv.org/abs/1506.01497) | Introduces the Region Proposal Network and the standard two-stage design. Essential for explaining what the proposal stage and the second stage each contribute. | |
| [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144) | Adds multi-scale feature maps to detectors. Explains why FPN levels matter for small objects and how proposals are assigned to levels. | |
| [Cascade R-CNN](https://arxiv.org/abs/1712.00726) | Refines boxes through several stages trained at increasing IoU thresholds. A strong upgrade over Faster R-CNN when localization quality limits mAP. | |
| [Libra R-CNN](https://arxiv.org/abs/1904.02701) | Addresses imbalance in RoI sampling, feature levels and the regression loss. Gives concrete ideas for the RoI sampling and box loss experiments. | |

## Models: one-stage anchor-based

| Resource | Description | Used in |
| --- | --- | --- |
| [SSD](https://arxiv.org/abs/1512.02325) | Predicts boxes from anchors at several feature map scales in a single pass. A simple baseline that shows the core one-stage idea. | |
| [RetinaNet / Focal Loss](https://arxiv.org/abs/1708.02002) | Shows that foreground-background imbalance is the main problem for one-stage detectors and fixes it with focal loss. Key reading for the question about removing the proposal stage. | |
| [YOLOv2](https://arxiv.org/abs/1612.08242) | Picks anchor shapes by k-means clustering of training boxes. Use this approach to ground your anchor configuration in the dataset statistics. || `01_eda.ipynb` §13; `src/models/anchors.py` |
| [ATSS](https://arxiv.org/abs/1912.02424) | Argues that the gap between anchor-based and anchor-free detectors comes mostly from how positives are assigned. Very useful for the comparisons across detector families. | |

## Models: one-stage anchor-free

| Resource | Description | Used in |
| --- | --- | --- |
| [FCOS](https://arxiv.org/abs/1904.01355) | Predicts boxes from every location inside an object, with centerness to suppress poor predictions. The clearest example of how locations are assigned to objects without anchors. | |
| [CenterNet (Objects as Points)](https://arxiv.org/abs/1904.07850) | Represents objects by their center points on a heatmap. Shows a different assignment scheme, which behaves differently for densely packed objects. | |
| [YOLOX](https://arxiv.org/abs/2107.08430) | An anchor-free YOLO with the SimOTA dynamic label assignment. Useful for the label assignment experiment. | |
| [Generalized Focal Loss](https://arxiv.org/abs/2006.04388) | Merges classification and localization quality into one score and models box edges as distributions. Its distribution focal loss is used in many recent YOLO versions. | |
| [TOOD](https://arxiv.org/abs/2108.07755) | Introduces task-aligned assignment, which picks positives by both classification score and IoU. Explains the assigner used in modern Ultralytics YOLOs. | |
| [YOLOv10](https://arxiv.org/abs/2405.14458) | Removes NMS from YOLO with consistent dual assignments. Relevant when comparing NMS-based detectors with query-based ones. | |
| [Ultralytics YOLO overview, YOLOv5 to YOLO26](https://arxiv.org/abs/2510.09653) | Summarizes architectural changes across Ultralytics releases. Check whether your chosen version uses NMS before comparing it with DETR-style models. | |

## Models: query-based

| Resource | Description | Used in |
| --- | --- | --- |
| [DETR](https://arxiv.org/abs/2005.12872) | Frames detection as set prediction with bipartite matching, removing anchors and NMS. Required reading for explaining how matching changes training. | |
| [Deformable DETR](https://arxiv.org/abs/2010.04159) | Replaces full attention with sparse deformable attention over multi-scale features. Fixes DETR's slow convergence and poor small-object performance. | |
| [DAB-DETR](https://arxiv.org/abs/2201.12329) | Uses anchor boxes as decoder queries. Helps interpret what queries represent and how their number relates to scene density. | |
| [DN-DETR](https://arxiv.org/abs/2203.01305) | Adds a denoising task that stabilizes bipartite matching early in training. The basis for the "denoising losses" tuning option. | |
| [DINO](https://arxiv.org/abs/2203.03605) | Combines denoising, mixed query selection and look-forward box refinement. A strong reference for what a well-tuned DETR looks like. | |
| [RT-DETR](https://arxiv.org/abs/2304.08069) and [RT-DETRv2](https://arxiv.org/abs/2407.17140) | Real-time DETRs with an efficient hybrid encoder. A practical choice for training a query-based model within the compute budget. | |
| [D-FINE](https://arxiv.org/abs/2410.13842) and [DEIM](https://arxiv.org/abs/2412.04234) | Recent real-time DETRs with improved box regression and faster convergence. Worth trying if RT-DETR does not converge within your budget. | |

## Frameworks and implementations

| Resource | Description | Used in |
| --- | --- | --- |
| [TorchVision detection tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html) and [reference scripts](https://github.com/pytorch/vision/tree/main/references/detection) | Show how to fine-tune Faster R-CNN, RetinaNet, FCOS and SSD on a custom dataset. The lowest-friction way to train anchor-based and FCOS models with full control over the code. | |
| [Detectron2](https://github.com/facebookresearch/detectron2) | Meta's detection library with configurable two-stage and RetinaNet models. Good if you want fine-grained control over RPN and RoI head settings. | |
| [MMDetection](https://github.com/open-mmlab/mmdetection) | The largest model zoo (Cascade R-CNN, ATSS, GFL, DINO and many more) under one config system. Pin library versions, since installation on recent PyTorch can be difficult. | |
| [Ultralytics](https://docs.ultralytics.com/) | Easy training of YOLO models and RT-DETR with a shared API, free under the AGPL-3.0 license. Convenient for keeping the training protocol identical across models. | |
| [Hugging Face object detection guide](https://huggingface.co/docs/transformers/tasks/object_detection) and [RT-DETRv2 docs](https://huggingface.co/docs/transformers/model_doc/rt_detr_v2) | Walk through fine-tuning DETR-family models with the Transformers API. Useful if you prefer the Hugging Face ecosystem for the query-based model. | |
| [Official RT-DETR code](https://github.com/lyuwenyu/RT-DETR) | The authors' implementation of RT-DETR and RT-DETRv2. Gives access to all training options, including the ones the higher-level wrappers hide. | |
| [PyTorch transfer learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) and [TorchVision models](https://docs.pytorch.org/vision/main/models.html) | Explain how to load pretrained weights and fine-tune them. These were in your original list and still fit here. | |

## Other sources

Sources outside the course list. Add a row whenever one is cited.

| Resource | Description | Used in |
| --- | --- | --- |
| [VisDrone toolkit README](https://github.com/VisDrone/VisDrone2018-VID-toolkit) | Meaning of the occlusion and truncation levels. | `01_eda.ipynb` §8 |
| [seaborn color palettes](https://seaborn.pydata.org/tutorial/color_palettes.html) | Source of the colorblind-safe class colors. | `src/visualization/style.py` |
| [torchvision detection transform](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/transform.py) | `GeneralizedRCNNTransform` resize rule (`min_size`, `max_size`). | `src/data/statistics.py`; `01_eda.ipynb` §5.4 |