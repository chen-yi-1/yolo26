from pathlib import Path

from scripts.prepare_yolo_dataset import prepare_yolo_dataset


def write_file(path, content=""):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_prepare_dataset_keeps_empty_label_backgrounds_with_class_sampling(tmp_path):
    source = tmp_path / "dataset"
    output = tmp_path / "datasets"
    write_file(source / "classes.txt", "abnormal\nhealthy\n")

    images = {
        "abnormal_a": source / "abnormal" / "abnormal_a.jpg",
        "healthy_a": source / "healthy" / "healthy_a.jpg",
        "background_a": source / "background" / "background_a.jpg",
    }
    for image_path in images.values():
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"fake image bytes")

    write_file(source / "labels" / "abnormal_a.txt", "0 0.5 0.5 0.2 0.2\n")
    write_file(source / "labels" / "healthy_a.txt", "1 0.5 0.5 0.2 0.2\n")
    write_file(source / "labels" / "background_a.txt", "")

    result = prepare_yolo_dataset(
        source,
        output,
        sample_count=1,
        task="detect",
        train_ratio=1.0,
    )

    assert result["train"] == 3
    assert (output / "images" / "train" / "background_a.jpg").is_file()
    assert (output / "labels" / "train" / "background_a.txt").is_file()
    assert (output / "labels" / "train" / "background_a.txt").read_text(encoding="utf-8") == ""


def test_prepare_dataset_limits_empty_label_backgrounds_when_requested(tmp_path):
    source = tmp_path / "dataset"
    output = tmp_path / "datasets"
    write_file(source / "classes.txt", "abnormal\nhealthy\n")

    image_paths = [
        source / "abnormal" / "abnormal_a.jpg",
        source / "healthy" / "healthy_a.jpg",
        source / "background" / "background_a.jpg",
        source / "background" / "background_b.jpg",
        source / "background" / "background_c.jpg",
    ]
    for image_path in image_paths:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"fake image bytes")

    write_file(source / "labels" / "abnormal_a.txt", "0 0.5 0.5 0.2 0.2\n")
    write_file(source / "labels" / "healthy_a.txt", "1 0.5 0.5 0.2 0.2\n")
    write_file(source / "labels" / "background_a.txt", "")
    write_file(source / "labels" / "background_b.txt", "")
    write_file(source / "labels" / "background_c.txt", "")

    result = prepare_yolo_dataset(
        source,
        output,
        sample_count=1,
        background_count=1,
        task="detect",
        train_ratio=1.0,
        seed=7,
    )

    copied_backgrounds = list((output / "labels" / "train").glob("background_*.txt"))
    assert result["train"] == 3
    assert len(copied_backgrounds) == 1


def test_prepare_dataset_limits_backgrounds_without_class_sampling(tmp_path):
    source = tmp_path / "dataset"
    output = tmp_path / "datasets"
    write_file(source / "classes.txt", "abnormal\n")

    for stem in ("abnormal_a", "background_a", "background_b"):
        folder = "abnormal" if stem.startswith("abnormal") else "background"
        image_path = source / folder / f"{stem}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"fake image bytes")

    write_file(source / "labels" / "abnormal_a.txt", "0 0.5 0.5 0.2 0.2\n")
    write_file(source / "labels" / "background_a.txt", "")
    write_file(source / "labels" / "background_b.txt", "")

    result = prepare_yolo_dataset(
        source,
        output,
        sample_count=0,
        background_count=1,
        task="detect",
        train_ratio=1.0,
    )

    copied_backgrounds = list((output / "labels" / "train").glob("background_*.txt"))
    assert result["train"] == 2
    assert len(copied_backgrounds) == 1
