\
from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class Sample:
    dataset: str
    source: str
    video_id: str
    frame_ids: List[int]
    image_paths: List[str]
    temporal_window: int
    meta: Dict[str, Any]
    tasks: Dict[str, Any]


def load_dataset(json_path: Path) -> List[Sample]:
    import json
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    samples: List[Sample] = []
    for item in data:
        samples.append(
            Sample(
                dataset=item.get("dataset", ""),
                source=item.get("source", ""),
                video_id=item["video_id"],
                frame_ids=item["frame_ids"],
                image_paths=item["image_paths"],
                temporal_window=item.get("temporal_window", 3),
                meta=item.get("meta", {}) or {},
                tasks=item["tasks"],
            )
        )
    return samples


def _encode_image_to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime = "image/png"
    if suffix in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"

    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def resolve_image_path(image_root: Path, rel_path: str) -> Path:
    '''
    Your JSON stores paths like:
      cholect50-challenge-val/videos/VID68/000016.png

    On Windows you said you have:
      .../dataset/cholect50-challenge-val/videos/VID68/000016.png

    But sometimes users pass image_root as:
      .../dataset/cholect50-challenge-val/videos

    So we try multiple strategies.
    '''
    p = Path(rel_path)

    candidates = [
        image_root / p,
        image_root / Path(*p.parts[1:]),  # strip leading "cholect50-challenge-val"
        image_root / Path(*p.parts[2:]),  # strip "cholect50-challenge-val/videos"
        image_root / p.name,              # last resort
    ]

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"Could not resolve image path. image_root={image_root} rel_path={rel_path}\n"
        f"Tried:\n" + "\n".join(str(x) for x in candidates)
    )


def load_images_as_data_urls(image_root: Path, image_paths: List[str]) -> List[str]:
    urls: List[str] = []
    for rp in image_paths:
        abs_path = resolve_image_path(image_root, rp)
        urls.append(_encode_image_to_data_url(abs_path))
    return urls
