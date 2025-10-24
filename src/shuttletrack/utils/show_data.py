from pathlib import Path
from argparse import ArgumentParser
import yaml
import random

from ultralytics.data.utils import visualize_image_annotations
from ultralytics.data.dataset import YOLODataset


def main():
    parser = ArgumentParser(description="Visualize some images with annotation")
    parser.add_argument("dataset", help="Path to dataset yaml file")
    parser.add_argument(
        "-n", "--number", help="number of files", required=False, default=1
    )
    args = parser.parse_args()
    path = Path(args.dataset)
    data_dict = yaml.safe_load(path.read_text())

    img_paths = [
        p
        for p in (path.parent / Path(data_dict["path"]) / data_dict["train"]).iterdir()
        if p.is_file()
    ]

    random.seed(1)
    img_paths_sample = random.sample(img_paths, int(args.number))

    for img_path in img_paths_sample:
        label_path = img_path.with_name(img_path.stem + ".txt")
        label_path = (
            label_path.parents[2] / "labels" / img_path.parts[-2] / label_path.name
        )

        assert label_path.exists()
        print(f"showing: {img_path}, with label {label_path}")

        visualize_image_annotations(img_path, label_path, {0: "shuttle"})


if __name__ == "__main__":
    main()
