# Research benchmarks

These scripts compare Project R against published models on the public SoccerNet benchmarks, using the official evaluation code. Every download goes inside the project folder: models in `.cache/`, data in `data/SoccerNet/` and results in `outputs/benchmarks/`. Nothing is written to the C: drive.

## 1. Install (once)

Run this from the project folder (`F:\AI_Projects\Project--R\Project--R`) with the project's `.venv` active:

```powershell
$env:PIP_CACHE_DIR = "$PWD\.cache\pip"; $env:TEMP = "$PWD\.cache\tmp"; $env:TMP = $env:TEMP
python benchmarks/setup_models.py --install
```

This installs `requirements-benchmarks.txt`. It also installs BoxMOT without its dependencies, because BoxMOT asks for OpenCV < 5 and would otherwise replace the project's OpenCV 5 and CUDA torch.

It then downloads every model, about 1.5 GB in total:

* Detectors: YOLO26 n/s/m/l/x, YOLO11x, RT-DETR l/x.
* ReID: OSNet ReID weights.
* Spotting: ResNet-152, NetVLAD++, and SoccerNet's sn-spotting repository (CALF weights, the PCA used for CALF features, and the official MaxPool/NetVLAD test predictions).
* OCR: the EasyOCR model, if EasyOCR is installed.

## 2. Data

```powershell
python benchmarks/download_data.py --login       # once: paste a Hugging Face read token
python benchmarks/download_data.py --spotting    # ~11 GB: 100 test matches, labels + ResNet features
python benchmarks/download_data.py --tracking    # ~10 GB zip + ~10 GB unzipped: 49 test clips
```

If the SoccerNet datasets need access, do this first:

1. Open <https://huggingface.co/SoccerNet> while logged in and accept the terms of SN-Labels, SN-Features and SN-Tracking-2023.
2. Create a read token at <https://huggingface.co/settings/tokens> and paste it into `--login`. The token is saved in `.cache/huggingface`.

You can delete the tracking zip after it is unzipped. The script tells you where it is.

## 3. Event spotting

```powershell
python benchmarks/spotting_benchmark.py                         # SoccerNet-v2 test set
python benchmarks/spotting_benchmark.py --video data/videos/test1.mp4   # your own clip
```

Methods, all with official SoccerNet weights or predictions:

| name | model | how it runs here |
|---|---|---|
| `maxpool` | MaxPool baseline (Deliège et al., CVPRW 2021) | official test predictions |
| `netvlad` | NetVLAD baseline (same paper) | official test predictions |
| `calf` | CALF (Cioppa et al., CVPR 2020) | official weights, official CALF code, PCA-512 features |
| `netvlad++` | NetVLAD++ (Giancola & Ghanem, CVSports 2021) | official weights through `src/action_spotting.py`, official test settings (30 s NMS) |
| `ours` | NetVLAD++ with the 10 s NMS of our live pipeline | shows what our setting costs on full matches |

The results go to `outputs/benchmarks/spotting_results.md` (and `.json`). They include average-mAP, loose (5 to 60 s tolerance) and tight (1 to 5 s), overall, for visible and unshown actions, and loose per class.

**Check first:** our numbers for the four official models should come out close to the published ones. The published loose average-mAP values are MaxPool 18.6, NetVLAD 31.4 and CALF 40.7 (SoccerNet-v2 paper) and NetVLAD++ 53.4 (NetVLAD++ paper). Check them against the papers before citing. If ours match, the setup is right, and only then is it worth comparing anything new.

**Newer published methods** have no weights released for these features, or need raw video and days of GPU training. Examples are E2E-Spot (Hong et al., ECCV 2022), Spivak (Soares et al., 2022), T-DEED (Xarles et al., CVPRW 2024), and the Baidu-feature models (Zhou et al., 2021). Cite their numbers from their papers or the SoccerNet leaderboard. Never cite numbers we did not reproduce or read in a paper.

On your own clip, `--video` lists what each model finds. With `--labels my_labels.json` it also scores them. That file uses the SoccerNet `Labels-v2.json` format with every annotation in half 1, and `position` in ms. Our scoreboard goals and close-up card detection need the raw broadcast video. SoccerNet's videos are under an NDA (<https://www.soccer-net.org/data>), so for the paper, label a few full matches of your own, or request the videos.

## 4. Player tracking

```powershell
# Step 1: every tracker on the same public detections, with and without our ID layer
python benchmarks/tracking_benchmark.py
# Step 2: every detector with our tracker
python benchmarks/tracking_benchmark.py --trackers botsort --detectors yolo26n yolo26s yolo26m yolo26l yolo26x yolo11x rtdetr-l rtdetr-x
# Quick look first (5 clips):
python benchmarks/tracking_benchmark.py --sequences 5 --trackers botsort bytetrack
```

Trackers:

* From Ultralytics: ByteTrack, and BoT-SORT with our `botsort_custom.yaml`.
* From BoxMOT: OC-SORT, Deep OC-SORT, StrongSORT, BoostTrack, Hybrid-SORT and BoT-SORT with ReID. These use OSNet appearance features.

Every run is done twice: with the tracker alone (`ID layer = none`), and with our stable-ID layer from `src/identity.py` on top (`ours`).

Metrics come from TrackEval, the code SoccerNet uses: HOTA, DetA, AssA, IDF1, MOTA and ID switches. The table also has tracker and detector speed. The ball is removed from the ground truth. The table is `outputs/benchmarks/tracking_results.md` (also `.csv` and `.json`), and it holds every finished run. Finished runs are reused; `--force` runs them again.

Time on a single GPU:

* Step 1 takes a few hours. The runs with our layer go at about 30 to 40 frames per second, and there are 36,750 frames.
* Step 2 adds about 10 to 30 minutes of detection per detector.

## Tests

`python -m pytest tests/test_benchmarks.py` runs the tracking benchmark end to end on a synthetic clip, and CALF on random features. It needs no SoccerNet download.

## Still to add

* **Shirt-number OCR:** EasyOCR against PARSeq and others on the SoccerNet jersey-number set.
* **Commentary:** our Bengali templates against an English LLM followed by translation (latency, wrong names), as in the paper roadmap.
