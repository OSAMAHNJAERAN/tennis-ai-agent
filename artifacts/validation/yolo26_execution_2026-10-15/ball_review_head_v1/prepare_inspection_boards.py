"""Exact nearest-neighbor views only; no detection or label acceptance."""
from pathlib import Path
import hashlib
import json
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
MANIFEST = ROOT / "outputs/ball_data_repair_review_wave96_v3/review_aids_manifest.json"
EXPECTED = "c6ec5bd28d0a36af85afb5f8018b136f9bc9afbbae65996bf9aa938ab2ab4a17"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if sha(MANIFEST) != EXPECTED or (OUT / "inspection_boards.json").exists():
        raise ValueError("Manifest mismatch or prior inspection receipt exists")
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    selected = document["proposals"]["positive_center_proposals"][:48]
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
    boards = []
    for begin in range(0, 48, 3):
        image = Image.new("RGB", (1260, 1074), "#172020")
        draw = ImageDraw.Draw(image)
        maps = []
        for row, item in enumerate(selected[begin:begin + 3]):
            y = row * 358
            draw.text((4, y + 2), f"Index {begin+row:02d} {item['proposal_id']} | {item['clip']} f{item['frame_index']} | {item['conservative_event_group']} {item['proposed_split']} | origin {item['native_patch_origin_xy_px']}", font=font, fill="white")
            for column, context in enumerate(("context0", "context1", "context2")):
                binding = item["images"][context]
                if sha(binding["path"]) != binding["sha256"]:
                    raise ValueError("Original patch changed")
                with Image.open(binding["path"]) as source:
                    source = source.convert("RGB")
                    if source.size != (96, 96):
                        raise ValueError("Unexpected source patch dimensions")
                    x = column * 288
                    image.paste(source.resize((288, 288), Image.Resampling.NEAREST), (x, y + 51))
                    draw.text((x + 4, y + 28), f"{context} unmarked 96x96 at3x", font=font, fill="#d2d5db")
                    maps.append({"id": item["proposal_id"], "context": context, "source": binding,
                                 "native_patch_origin_xy_px": item["native_patch_origin_xy_px"], "source_patch_xyxy": [0, 0, 96, 96],
                                 "board_xyxy": [x, y + 51, x + 288, y + 339], "scale": 3})
                    if context == "context1":
                        zoom = source.crop((32, 32, 64, 64)).resize((384, 384), Image.Resampling.NEAREST)
                        # Fit the zoom to the same row height at8x rather than
                        # smooth or interpolate its observed native pixels.
                        zoom = source.crop((30, 30, 66, 66)).resize((288, 288), Image.Resampling.NEAREST)
                        image.paste(zoom, (972, y + 51))
                        draw.text((874, y + 28), "context1 crop[30:66] at8x", font=font, fill="#ffdb7a")
                        for tick in range(30, 66, 4):
                            draw.text((972 + (tick - 30) * 8, y + 339), str(tick), font=font, fill="#ffdb7a")
                        maps.append({"id": item["proposal_id"], "context": "context1", "source": binding,
                                     "native_patch_origin_xy_px": item["native_patch_origin_xy_px"], "source_patch_xyxy": [30, 30, 66, 66],
                                     "board_xyxy": [972, y + 51, 1260, y + 339], "scale": 8})
        path = OUT / f"native_unmarked_{begin:02d}_{begin+2:02d}.png"
        image.save(path)
        boards.append({"path": str(path), "sha256": sha(path), "kind": "UNMARKED_NATIVE_PATCH_AND_NEAREST_ZOOM", "maps": maps})
    for begin in range(0, 48, 12):
        image = Image.new("RGB", (1440, 1240), "#172020")
        draw = ImageDraw.Draw(image)
        maps = []
        for offset, item in enumerate(selected[begin:begin + 12]):
            x, y = offset % 3 * 480, offset // 3 * 310
            binding = item["images"]["native_frame"]
            if sha(binding["path"]) != binding["sha256"]:
                raise ValueError("Original native frame changed")
            with Image.open(binding["path"]) as source:
                source = source.convert("RGB")
                image.paste(source.resize((480, 270), Image.Resampling.NEAREST), (x, y + 40))
                maps.append({"id": item["proposal_id"], "source": binding, "original_size": list(source.size),
                             "board_xyxy": [x, y + 40, x + 480, y + 310], "resize": "NEAREST_FOR_VIEW_SCOPE_ONLY"})
            draw.text((x + 4, y + 2), f"{begin+offset:02d} {item['proposal_id']} {item['clip']}", font=font, fill="white")
            draw.text((x + 4, y + 21), f"f{item['frame_index']} {item['conservative_event_group']}", font=font, fill="#ffdb7a")
        path = OUT / f"full_frame_scope_{begin:02d}_{begin+11:02d}.png"
        image.save(path)
        boards.append({"path": str(path), "sha256": sha(path), "kind": "COMPLETE_NATIVE_FRAME_OVERVIEW", "maps": maps})
    receipt = {"manifest": str(MANIFEST), "manifest_sha256": EXPECTED, "selection_indices": [0, 47],
               "selection_ids": [item["proposal_id"] for item in selected], "script_sha256": sha(__file__),
               "resampling": "NEAREST_ONLY_NO_MARKS_ON_PIXEL_PATCHES_NO_DETECTION_NO_BOX_DECISIONS", "boards": boards}
    (OUT / "inspection_boards.json").write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"boards": len(boards), "manifest_sha256": EXPECTED, "output": str(OUT)}))


if __name__ == "__main__":
    main()
