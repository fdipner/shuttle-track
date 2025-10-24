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

FOLDERS = ["images", "labels"]
SUBFOLDERS = ["train", "val"]
FORMAT_STRING = "09d"


@dataclass
class Label:
    x: float
    y: float
    width: float
    height: float
    rotation: float
    frame: int
    enabled: bool
    time: float
    in_yolo_format: bool = False

    def convert_to_yolo(self) -> None:
        self.in_yolo_format = True
        # from precentages to (0,1)
        self.x /= 100.0
        self.y /= 100.0
        self.width /= 100.0
        self.height /= 100.0

        # from top left to center
        self.x += self.width / 2.0
        self.y += self.height / 2.0

    def assert_yolo_format(self) -> None:
        if not self.in_yolo_format:
            self.convert_to_yolo()

    def get_yolo_string(self, object_id: int = 0) -> str:
        self.assert_yolo_format()

        return f"{object_id} {self.x} {self.y} {self.width} {self.height}\n"


def interpolate_labels(label1: Label, label2: Label, at_frame: int) -> Label:
    assert label1.in_yolo_format == label2.in_yolo_format

    x = np.interp(at_frame, [label1.frame, label2.frame], [label1.x, label2.x])
    y = np.interp(at_frame, [label1.frame, label2.frame], [label1.y, label2.y])
    width = np.interp(
        at_frame, [label1.frame, label2.frame], [label1.width, label2.width]
    )
    height = np.interp(
        at_frame, [label1.frame, label2.frame], [label1.height, label2.height]
    )
    rotation = np.interp(
        at_frame, [label1.frame, label2.frame], [label1.rotation, label2.rotation]
    )
    time = np.interp(at_frame, [label1.frame, label2.frame], [label1.time, label2.time])

    return Label(x, y, width, height, rotation, at_frame, False, time, True)


def write_label_file(label_path: Path, video_name: str, frame_number: int, label: str):
    label_file = label_path / f"{video_name}_{format(frame_number,FORMAT_STRING)}.txt"
    label_file.write_text(label)


def convert_labels(annotation_file: dict, dir: Path, validation_set: list[Path]):
    video_rel_path = Path(annotation_file["data"]["video"])
    video_path = Path(video_rel_path.name)

    tqdm.write(f"processing: {video_path}")
    subpath = "val" if video_path in validation_set else "train"
    img_path = dir / "images" / subpath
    label_path = dir / "labels" / subpath

    # annotation
    annotation = annotation_file["annotations"][0]["result"][0]["value"]

    labels = annotation["sequence"]

    previous_label = Label(**labels[0])
    previous_label.convert_to_yolo()

    # fill up empty start frames
    for i in range(1, previous_label.frame):
        write_label_file(label_path, video_path.stem, i, "")

    vid_fps, vid_n_frames = get_video_info(video_path)

    assert (
        vid_n_frames <= annotation["framesCount"]
    )  # due to wired labelstudio bug that at times takes the last frame twice

    for label_dict in labels[:vid_n_frames]:
        label = Label(**label_dict)
        label.convert_to_yolo()

        write_label_file(
            label_path, video_path.stem, label.frame, label.get_yolo_string()
        )

        for i in range(previous_label.frame + 1, label.frame):
            if not previous_label.enabled:
                write_label_file(label_path, video_path.stem, i, "")
            else:
                interp = interpolate_labels(previous_label, label, i)
                write_label_file(
                    label_path,
                    video_path.stem,
                    i,
                    interp.get_yolo_string(),
                )
        previous_label = label

    for i in range(previous_label.frame + 1, vid_n_frames + 1):
        if not previous_label.enabled:
            write_label_file(label_path, video_path.stem, i, "")
        else:
            write_label_file(
                label_path, video_path.stem, i, previous_label.get_yolo_string()
            )

    # video
    label_fps = previous_label.frame / previous_label.time
    assert np.allclose(
        vid_fps, label_fps
    ), f"framerates did not match {label_fps=}, {vid_fps=}"

    ffmpeg.input(str(video_path)).output(
        str(img_path / f"{video_path.stem}_%{FORMAT_STRING}.jpg"), **{"q:v": 1}
    ).run(quiet=True)


def get_video_info(path: Path) -> tuple[float, int]:
    probe = ffmpeg.probe(path)
    video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
    fps_str = video_info["avg_frame_rate"]
    num, den = map(int, fps_str.split("/"))
    fps = num / den

    num_frames = int(video_info["nb_frames"])

    return fps, num_frames


def create_folders(path: Path) -> None:
    path.mkdir(exist_ok=True)
    for folder, subfolder in product(FOLDERS, SUBFOLDERS):
        (path / folder / subfolder).mkdir(parents=True, exist_ok=True)


def create_yaml(path: Path, objects: dict[int, str]):
    dataset = {
        "path": str(path.name),
        "train": "images/train",
        "val": "images/val",
        "names": objects,
    }
    yaml_path = path.parent / f"{str(path.name)}.yaml"
    print(yaml_path)
    yaml_path.write_text(yaml.safe_dump(dataset))


def get_argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="convert data form labelstudio to yolo format"
    )
    parser.add_argument("label", help="labels in labelstud.io format")
    parser.add_argument("out_dir", help="Path to output data")
    parser.add_argument(
        "-v",
        "--validation",
        action="append",
        help="videos wich should go into validation set",
    )

    return parser.parse_args()


def main():
    args = get_argparser()
    dir = Path(args.out_dir).resolve()

    create_folders(dir)
    create_yaml(dir, {0: "shuttle"})

    labels = json.loads(Path(args.label).read_text())
    validation_files = [Path(arg) for arg in args.validation]
    for video_file in tqdm(labels):
        convert_labels(video_file, dir, validation_files)


if __name__ == "__main__":
    main()
