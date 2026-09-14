"""Download a bounded, revision-pinned tennis validation subset with provenance."""

import argparse
import hashlib
import json
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/external/racketvision_validation"))
    parser.add_argument("--clips", type=int, default=6)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--revision", help="Pin a previously reviewed publisher dataset revision")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory to preserve acquired evidence")
    if not 1 <= args.clips <= 38 or args.start_index < 0:
        raise ValueError("Choose 1–38 clips and a nonnegative start index")
    repo = "linfeng302/RacketVision"
    session = requests.Session()

    def get(url):
        response = session.get(url, timeout=60)
        response.raise_for_status()
        return response

    revision = args.revision or get(f"https://huggingface.co/api/datasets/{repo}").json()["sha"]
    base = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/"
    card = get(base + "README.md").text
    if "license: mit" not in card[:1000].lower():
        raise ValueError("Dataset card license changed; review before acquisition")
    validation = get(base + "tennis/info/val.json").json()
    train = get(base + "tennis/info/train.json").json()
    selected = (train if args.split == 'train' else validation)[args.start_index:args.start_index + args.clips]
    if len(selected) != args.clips:
        raise ValueError('Selection extends beyond publisher split')
    train_matches = {item[0] for item in train}
    validation_matches = {item[0] for item in validation}
    if train_matches & validation_matches:
        raise ValueError("Selected validation match IDs overlap publisher training split")
    videos = get(f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/tennis/videos?limit=1000").json()
    metadata = {entry["path"]: entry for entry in videos if entry["type"] == "file"}
    paths = [f"tennis/videos/{match}_{rally}.mp4" for match, rally in selected]
    if sum(metadata[path]["size"] for path in paths) > 200_000_000:
        raise ValueError("Selected clips exceed the 200 MB acquisition budget")
    args.output.mkdir(parents=True)
    (args.output / "DATASET_CARD.md").write_text(card, encoding="utf-8")
    files = []

    def save(path, expected=None):
        content = get(base + path).content
        digest = hashlib.sha256(content).hexdigest()
        if expected and (len(content) != expected["size"] or
                         digest != expected.get("lfs", {}).get("oid")):
            raise ValueError(f"Publisher checksum mismatch: {path}")
        dest = args.output / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        files.append({"path": path, "bytes": len(content), "sha256": digest, "url": base + path})
        print(f"Acquired {path}: {len(content)} bytes", flush=True)

    save("tennis/info/val.json")
    save("tennis/info/train.json")
    if args.split == 'val':
        save("tennis/info/val_coco.json")
    for match, rally in selected:
        path = f"tennis/videos/{match}_{rally}.mp4"
        save(path, metadata[path])
        save(f"tennis/all/{match}/csv/{rally}_ball.csv")
    manifest = {
        "schema_version": "1.0", "source": repo, "revision": revision,
        "declared_license": "MIT (publisher dataset card)",
        "selection": f"Publisher {args.split} entries {args.start_index}:{args.start_index + args.clips}; chosen before inference",
        "split": "TRAINING_ONLY" if args.split == 'train' else "VALIDATION_ONLY", "selected_clips": selected,
        "publisher_train_match_id_overlap": [],
        "source_broadcast_independence": "UNVERIFIED; match IDs may denote clips",
        "existing_yolo_training_overlap": "UNVERIFIED",
        "final_test_downloaded": False, "qualification_evidence": False,
        "ground_truth_source": "Publisher raw ball CSV; val COCO racket annotations only for validation; no interpolated predictions",
        "ball_annotation_resolution": [1920, 1080],
        "dataset_card_sha256": hashlib.sha256(card.encode()).hexdigest(), "files": files,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
