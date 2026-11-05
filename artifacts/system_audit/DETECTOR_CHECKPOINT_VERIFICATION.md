# SmartPed AI Detector Checkpoint and Training History Verification

**Investigation date:** 2026-11-05  
**Scope:** Read-only inspection of `.pt` files, YOLO training run arguments/results/plots, inference source and configuration, and saved video-run metadata. No training, inference, model download, or existing-file modification was performed.

## Executive conclusion

The previous audit's detector-training conclusion was incomplete. It inspected the initial `runs/detect/train` run, but the project also contains subsequent runs `train-3`, `train-4`, and `train-5`. These later runs have saved PIE fine-tuned checkpoint files and training plots. In particular, all four previously reported metric values are present in `runs/detect/train-5/results.csv` at epoch 11:

| Metric | Saved value | Reported rounded value |
|---|---:|---:|
| Precision | 0.35896 | 0.359 |
| Recall | 0.32509 | 0.325 |
| mAP50 | 0.25915 | 0.259 |
| mAP50-95 | 0.13807 | 0.138 |

These are per-epoch validation metrics from the PIE **validation** split, not held-out test-set metrics. The training history has a `runs/detect/train-5/weights/best.pt` file, but the saved `results.csv` does not itself prove the internal epoch stored in that checkpoint. The checkpoint was not deserialized or run, so runtime loadability is not certified by this read-only investigation.

The current pedestrian video adapter, vehicle pipeline configuration, and end-to-end video runner use the project-root `yolo11s.pt` by default. The saved 60-second run records that same path as its shared detector checkpoint. It did **not** record a `train-5` checkpoint path or a checkpoint hash. The latest render-v2 is a rendering-only derivative of that saved run, not a new inference run.

## Checkpoint inventory

The project-tree search found the following seven `.pt` files. Hashes are SHA-256; modification times are local time (UTC+04:00).

| File | Size (bytes) | Last modified (UTC+04:00) | SHA-256 | Provenance assessment |
|---|---:|---|---|---|
| [yolo11s.pt](../../yolo11s.pt) | 19,313,732 | 2026-11-02 19:39:59 | `85A76FE86DD8AFE384648546B56A7A78580C7CB7B404FC595F97969322D502D5` | Appears to be the pretrained YOLO11s source checkpoint: training args name `yolo11s.pt` with `pretrained: true`, vehicle experiment report calls it pretrained, and all inference defaults point to it. Its original publisher/provenance is not independently attested by these files. |
| [train-3 best.pt](../../runs/detect/train-3/weights/best.pt) | 19,147,290 | 2026-11-02 22:20:21 | `B6287C2F8774A9F150DD3F45C1DE1C943E4E4390D2CAB273B6763148D66B1E0C` | Appears PIE fine-tuned: run args specify PIE data and initialize from `yolo11s.pt` with `pretrained: true`. |
| [train-3 last.pt](../../runs/detect/train-3/weights/last.pt) | 19,147,290 | 2026-11-02 22:20:21 | `F294F7C0123AE4E4827989250737F2AFEE14860D4E56361DDFEB232D062A9258` | Appears PIE fine-tuned; final saved epoch of this one-epoch run. |
| [train-4 best.pt](../../runs/detect/train-4/weights/best.pt) | 19,148,506 | 2026-11-02 23:27:46 | `E8FACCDB9A31C145B0A7119E93B6EA996EAA7433C71C828A226B7117EC5F858D` | Appears PIE fine-tuned: run args specify PIE data and initialize from `yolo11s.pt` with `pretrained: true`. |
| [train-4 last.pt](../../runs/detect/train-4/weights/last.pt) | 19,148,506 | 2026-11-02 23:27:46 | `148125E83EE5523EC9BD4C7E46047482609B7C34B80C312BF0CDB0ECEFB806D8` | Appears PIE fine-tuned; final saved epoch of this ten-epoch run. |
| [train-5 best.pt](../../runs/detect/train-5/weights/best.pt) | 19,149,914 | 2026-11-04 07:36:39 | `9E1D876FF9CCC49D03F43DA9DBE5BB4DF71C2A7514DBAAD24585E4A1F855ECFB` | Appears PIE fine-tuned from a resumed training run. Candidate best checkpoint for the verified epoch-11 metric row, but internal checkpoint epoch was not independently inspected. |
| [train-5 last.pt](../../runs/detect/train-5/weights/last.pt) | 19,149,978 | 2026-11-04 07:36:39 | `46B023B309AE04E30A140F225ABB0EC911E0BFC7555024A855F6F442E764950C` | Appears PIE fine-tuned; corresponds to the latest history row, epoch 21. |

No `best.pt` or `last.pt` was found under `train` or `train-2`. The inventory establishes that the listed weight files exist and their bytes were readable for hashing; it does not certify that a loader can deserialize them. No checkpoint was loaded or used for inference.

## Training-run comparison

All five run directories have an `args.yaml`. The dataset configuration identifies one pedestrian class, with `images/train`, `images/validation`, and `images/test`; the splits are assigned by recording set. Runs 1–4 specify `artifacts/pie_detection_yolo/data.yaml`, image size 640, and device `"0"`. `train-5` uses the same dataset and image size/device.

| Run | Initialization / resume evidence | Configured epochs; results epochs actually recorded | Batch | Saved weights | Results / plots |
|---|---|---:|---:|---|---|
| `train` | `model: yolo11s.pt`, `pretrained: true`, `resume: false` | 1; no `results.csv` | 4 | Missing | `args.yaml` only among inspected run outputs; results and plots missing. |
| `train-2` | `model: yolo11s.pt`, `pretrained: true`, `resume: false` | 1; no `results.csv` | 4 | Missing | `args.yaml` only among inspected run outputs; results and plots missing. |
| `train-3` | `model: yolo11s.pt`, `pretrained: true`, `resume: false` | 1; 1 | 4 | `best.pt`, `last.pt` | `results.csv` and validation plots exist. |
| `train-4` | `model: yolo11s.pt`, `pretrained: true`, `resume: false` | 10; 10 | 8 | `best.pt`, `last.pt` | `results.csv` and validation plots exist. |
| `train-5` | `model: runs\detect\train-5\weights\last.pt`, `resume: runs\detect\train-5\weights\last.pt`, `pretrained: true` | 30; 21 | 8 | `best.pt`, `last.pt` | `results.csv` and validation plots exist; the last recorded row is epoch 21, not the configured 30. |

`train-5` records AdamW, learning rate `lr0: 0.002`, and a resume path pointing to its own `last.pt`. This supports that it is a resumed/fine-tuning run, but the available arguments do not fully establish the checkpoint's earlier ancestry before the recorded resume. The cause of the 21-of-30 recorded epochs is not documented in these files.

### Results recorded in each run

The CSV columns are validation precision, recall, mAP50, and mAP50-95 for the run's validation split. The first four metrics below are reported at the last recorded epoch for each run, except the separately highlighted train-5 epoch-11 tuple.

| Run / epoch | Precision | Recall | mAP50 | mAP50-95 | Interpretation |
|---|---:|---:|---:|---:|---|
| `train-3`, epoch 1 | 0.26278 | 0.20116 | 0.13498 | 0.05532 | One-epoch validation row. |
| `train-4`, epoch 10 | 0.31845 | 0.28161 | 0.23227 | 0.12033 | Final row of ten recorded epochs. |
| `train-5`, epoch 11 | **0.35896** | **0.32509** | **0.25915** | **0.13807** | Exact source of the previously reported rounded values. |
| `train-5`, epoch 21 | 0.37631 | 0.29563 | 0.24390 | 0.12781 | Latest row present in `results.csv`; not the same metric tuple as epoch 11. |

`train-4/results.csv` and `train-5/results.csv` contain all recorded rows noted above; `train-3/results.csv` contains one. The training plots, including `results.png`, `BoxPR_curve.png`, `BoxF1_curve.png`, `BoxP_curve.png`, `BoxR_curve.png`, and confusion matrices, exist for `train-3`, `train-4`, and `train-5`. These plots do not replace the missing metric files for `train` and `train-2`.

The available plots were visually inspected. `train-3` has only one epoch, so its plots cannot establish a learning trend. `train-4` shows decreasing training/validation losses over its ten epochs, while precision, recall, and mAP fluctuate. `train-5` shows improving training losses, but validation classification loss turns upward after roughly epoch 11 and mAP metrics fluctuate rather than improve consistently through epoch 21. Its mAP50-95 is highest at epoch 11 in the saved CSV; mAP50 is marginally highest at epoch 12. The confusion-matrix plots show many detections in the background column at the displayed evaluation threshold; because PIE labels are selective, these counts cannot automatically be interpreted as confirmed false positives.

The recorded precision/recall values depend on the validation evaluation settings. In the saved training arguments, `conf` is null (framework default) and `iou` is 0.7. They should not be represented as measurements at the inference pipeline's confidence threshold of 0.25.

## Verification of the previously reported metrics

**Verified:** The four rounded values match a single row in [`train-5/results.csv`](../../runs/detect/train-5/results.csv), epoch 11: `0.35896`, `0.32509`, `0.25915`, `0.13807`.

**Not verified by these numbers:** A held-out test result, the performance of the current production/default checkpoint, or detector performance on all PIE objects. The run args set `split: val`; the metrics are validation metrics, not set04/test-split metrics.

**Checkpoint association:** The exact run and metric epoch are verified from the CSV. `train-5/weights/best.pt` exists and is the run's designated best checkpoint. The checkpoint's internal epoch and its correspondence to epoch 11 were not independently verified; no claim that this checkpoint contains exactly epoch 11 is made. The `last.pt` file belongs to the final saved history state, epoch 21, as indicated by the run record, but was not loaded.

## Source-code and configuration checkpoint references

- **Pedestrian detection/tracking adapter:** [`smartped_ai/trajectory/readers.py`](../../smartped_ai/trajectory/readers.py) instantiates `YOLO(str(model_path))` for video input. [`smartped_ai/trajectory/cli.py`](../../smartped_ai/trajectory/cli.py) sets its default to project-root `yolo11s.pt` and permits a `--model` override.
- **Vehicle detection/tracking:** [`configs/vehicle_tracking.json`](../../configs/vehicle_tracking.json) sets `model_path` to `../yolo11s.pt`; [`smartped_ai/vehicle_tracking/pipeline.py`](../../smartped_ai/vehicle_tracking/pipeline.py) loads that configured path. The saved vehicle experiment metadata also records the root `yolo11s.pt` path and identifies it as pretrained.
- **End-to-end runner:** [`smartped_ai/video_runner/pipeline.py`](../../smartped_ai/video_runner/pipeline.py) defines project-root `yolo11s.pt` as `_DEFAULT_MODEL`, exposes `--model`, and passes the same `model_path` to pedestrian extraction and the generated vehicle config. Its stages are sequential; each existing detector pipeline instantiates the shared checkpoint separately.
- **Other detector-loading source found:** The search found the same video detector adapter in trajectory extraction and the vehicle pipeline. The crossing-intention Random Forest uses a separate `.pkl` artifact, not YOLO weights.
- **Fine-tuned checkpoint defaults:** No inspected inference source or active inference configuration defaults to `runs/detect/train-3`, `train-4`, or `train-5`. Those checkpoints require an explicit model-path override.

## Provenance of the latest annotated video

The saved end-to-end run is [`processing_metrics.json`](../video_runs/set01_video_0001_runner_60s/processing_metrics.json). It records:

- `shared_detector_checkpoint`: `C:\Users\vlctus\OneDrive\Desktop\Pattern\project development\yolo11s.pt`
- vehicle experiment `model_path`: the same project-root `yolo11s.pt`
- two sequential model loads from the existing pedestrian and vehicle pipelines
- 1,800 processed frames, 30 FPS, 1920×1080, first 60 seconds of `set01/video_0001`

The latest annotated-video render is [`render_metrics.json`](../video_runs/set01_video_0001_render_v2/render_metrics.json). It records the previous run directory as `source_run_dir`; render-v2 reuses that run's detections/interaction CSVs and does not perform new detector inference. Therefore its detector provenance is the original run's project-root `yolo11s.pt`, not `train-5/best.pt`.

The saved run records a path, not a hash. The current `yolo11s.pt` SHA-256 is listed in the inventory, but there is no run-time hash in the metrics to prove byte-for-byte that the file at that path was identical when the video was generated. The training outputs for `train-3` through `train-5` are in separate run directories, so they are not the path the runner records.

## Explicit answers

1. **Does a usable PIE-fine-tuned pedestrian checkpoint exist?** PIE fine-tuned checkpoint artifacts do exist, including `train-5/weights/best.pt` and `last.pt`, as well as `train-3` and `train-4` checkpoints. Their training args/results provide evidence they are fine-tuned artifacts. However, the investigation cannot certify that they are operationally usable: under the read-only/no-inference restriction, they were not deserialized or executed.
2. **Are the previously reported training metrics verifiable?** Yes. All four values are in `train-5/results.csv` at epoch 11 and round to the cited figures. They are validation metrics, not held-out test metrics.
3. **Which exact checkpoint was used in the latest annotated video?** The saved inference run records the project-root `yolo11s.pt`. The latest render-v2 derives from that run and adds no inference. The saved run does not record a checkpoint hash.
4. **Are pedestrian and vehicle pipelines using the same checkpoint?** Yes, in the end-to-end runner: the same `model_path` is passed to both stages. The run metrics explicitly identify the shared root `yolo11s.pt`. Each pipeline loads it separately and sequentially.
5. **Did the previous audit overlook an existing training run or checkpoint?** Yes. The audit's statement that no fine-tuned checkpoint was found was too broad: it inspected the initial `train` run but missed `train-3`, `train-4`, and `train-5`, which contain results, plots, and saved weights. The initial `train` and `train-2` runs still have no recorded results or weight files in the inspected tree.
6. **What is the correct next step?** In a separately approved evaluation (not performed here), evaluate `train-5/weights/best.pt` on the reserved PIE test split using recorded, explicit evaluation settings; compare it with the current pretrained `yolo11s.pt` baseline; save the checkpoint hash, exact evaluation configuration, and results; and confirm runtime loadability as part of that evaluation. Account for PIE's sparse/selective labels when interpreting precision and unmatched detections. Do not silently switch the video runner's default checkpoint based on validation metrics.
