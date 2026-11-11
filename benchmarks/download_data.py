"""
Download the SoccerNet data the benchmarks need, into data/SoccerNet
inside the project (never the C: drive: the Hugging Face cache is in
.cache/huggingface, see project_cache.py).

    python benchmarks/download_data.py --login       # once, if asked to
    python benchmarks/download_data.py --spotting    # ~11 GB
    python benchmarks/download_data.py --tracking    # ~10 GB zip + ~10 GB unzipped

--spotting: the 100 SoccerNet-v2 test matches: action labels
  (Labels-v2.json) and the ResNet-152 features the models were trained
  on, full (ResNET_TF2, 2048-d, for NetVLAD++) and PCA-512 (for CALF).
--tracking: SoccerNet-Tracking 2023 test split: 49 thirty-second clips,
  frames + ground truth + public detections.

The data comes from the SoccerNet datasets on Hugging Face. If a download
says you have no access: open https://huggingface.co/SoccerNet, accept
the dataset terms while logged in, then run --login with a read token
from https://huggingface.co/settings/tokens.
"""

import argparse
import zipfile

from common import SOCCERNET

SPOTTING_FILES = ["Labels-v2.json",
                  "1_ResNET_TF2.npy", "2_ResNET_TF2.npy",
                  "1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"]


def login():
    from huggingface_hub import login as hf_login
    hf_login()  # the token is saved under .cache/huggingface


def download_spotting(downloader, with_full_features=True):
    files = SPOTTING_FILES if with_full_features else \
        [f for f in SPOTTING_FILES if "PCA" in f or f.startswith("Labels")]
    downloader.downloadGames(files=files, split=["test"], task="spotting")


def download_tracking(downloader, delete_zip=False):
    folder = SOCCERNET / "tracking-2023"
    downloader.downloadDataTask(task="tracking-2023", split=["test"])
    _unzip(folder / "test.zip", delete_zip)
    if _missing_ground_truth(folder / "test"):
        # Ground truth shipped separately: fetch it too.
        downloader.downloadDataTask(task="tracking-2023", split=["test_labels"])
        _unzip(folder / "test2023-private.zip", delete_zip)
    missing = _missing_ground_truth(folder / "test")
    if missing:
        print(f"Warning: {len(missing)} clips have no gt/gt.txt, e.g. "
              f"{missing[0]}. The tracking benchmark skips them.")


def _missing_ground_truth(test_folder):
    if not test_folder.exists():
        return ["(no test folder)"]
    return [p.name for p in sorted(test_folder.iterdir())
            if p.is_dir() and not (p / "gt" / "gt.txt").exists()]


def _unzip(archive, delete_zip):
    if not archive.exists():
        return
    print(f"Unzipping {archive}")
    with zipfile.ZipFile(archive) as file:
        file.extractall(archive.parent)
    if delete_zip:
        archive.unlink()
    else:
        print(f"You can delete {archive} now to free space "
              "(or rerun with --delete-zip).")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--login", action="store_true",
                        help="save a Hugging Face token (inside the project)")
    parser.add_argument("--spotting", action="store_true")
    parser.add_argument("--tracking", action="store_true")
    parser.add_argument("--no-full-features", action="store_true",
                        help="--spotting without the 2048-d features (~9 GB "
                             "less, but then NetVLAD++ cannot run)")
    parser.add_argument("--delete-zip", action="store_true",
                        help="delete the tracking zip after unzipping it")
    args = parser.parse_args()
    if not (args.login or args.spotting or args.tracking):
        parser.error("choose --login, --spotting and/or --tracking")

    if args.login:
        login()
    from SoccerNet.Downloader import SoccerNetDownloader
    SOCCERNET.mkdir(parents=True, exist_ok=True)
    downloader = SoccerNetDownloader(LocalDirectory=str(SOCCERNET))
    if args.spotting:
        download_spotting(downloader, not args.no_full_features)
    if args.tracking:
        download_tracking(downloader, args.delete_zip)
    print("Data folder:", SOCCERNET)


if __name__ == "__main__":
    main()
