from itertools import product
import argparse
from pathlib import Path
import json
import yaml
import numpy as np
from dataclasses import dataclass

# import cv2
import ffmpeg
from tqdm import tqdm


def get_video_info(path: Path) -> tuple[float, int]:
    probe = ffmpeg.probe(path)
    video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
    fps_str = video_info["avg_frame_rate"]
    num, den = map(int, fps_str.split("/"))
    fps = num / den

    num_frames = int(video_info["nb_frames"])

    return fps, num_frames


def convert_labels(annotation_file: dict):
    video_path = Path(annotation_file["file_upload"])
    tqdm.write(f"processing: {video_path}")

    # annotation
    annotation = annotation_file["annotations"][0]["result"][0]["value"]

    labels = annotation["sequence"]

    fps_vid, num_frames_vid = get_video_info(video_path)
    annotation["framesCount"] = num_frames_vid

    for label_dict in labels:
        time = label_dict["time"]
        # fps = frames / second -> frame_num = fps*seconds
        label_dict["frame"] = int(fps_vid * time)
        # fps = frames / second -> seconds = frames/fps
        label_dict["time"] = label_dict["frame"] / fps_vid


def get_argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="convert framerate in labelstudio")
    parser.add_argument("label", help="labels in labelstud.io format")
    parser.add_argument("out_dir", help="Path to output data")
    return parser.parse_args()


def main():
    args = get_argparser()
    dir = Path(args.out_dir).resolve()

    dir.mkdir(exist_ok=True)
    label_path = Path(args.label)
    labels = json.loads(label_path.read_text())
    for video_file in tqdm(labels):
        convert_labels(video_file)

    out_file = dir / label_path.name

    out_file.write_text(json.dumps(labels))


if __name__ == "__main__":
    main()
