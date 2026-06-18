from train import (
    build_train_kwargs,
)
from utils.training_plots import (
    add_training_plot_callback,
    plot_training_results,
    plot_training_results_from_run,
)


def test_build_train_kwargs_includes_augmentation_parameters():
    kwargs = build_train_kwargs(
        data_yaml="datasets/datasets.yaml",
        task="segment",
        imgsz=640,
        epochs=100,
        batch=16,
        device=0,
        workers=4,
        project=None,
        name="train",
        exist_ok=False,
        pretrained=True,
        resume=False,
        optimizer="auto",
        lr0=0.001,
        patience=20,
        save_period=10,
        amp=True,
        freeze=None,
        cache=False,
        plots=True,
        val=True,
        verbose=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.2,
        scale=0.6,
        shear=2.0,
        perspective=0.001,
        flipud=0.1,
        fliplr=0.5,
        bgr=0.0,
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.3,
        copy_paste_mode="flip",
        auto_augment="randaugment",
        erasing=0.4,
        crop_fraction=1.0,
        close_mosaic=10,
        class_weights=None,
    )

    assert kwargs["hsv_h"] == 0.015
    assert kwargs["hsv_s"] == 0.7
    assert kwargs["hsv_v"] == 0.4
    assert kwargs["degrees"] == 5.0
    assert kwargs["translate"] == 0.2
    assert kwargs["scale"] == 0.6
    assert kwargs["shear"] == 2.0
    assert kwargs["perspective"] == 0.001
    assert kwargs["flipud"] == 0.1
    assert kwargs["fliplr"] == 0.5
    assert kwargs["bgr"] == 0.0
    assert kwargs["mosaic"] == 1.0
    assert kwargs["mixup"] == 0.2
    assert kwargs["copy_paste"] == 0.3
    assert kwargs["copy_paste_mode"] == "flip"
    assert kwargs["auto_augment"] == "randaugment"
    assert kwargs["erasing"] == 0.4
    assert kwargs["crop_fraction"] == 1.0
    assert kwargs["close_mosaic"] == 10


def test_build_train_kwargs_rejects_class_weights():
    try:
        build_train_kwargs(
            data_yaml="datasets/datasets.yaml",
            task="detect",
            imgsz=640,
            epochs=100,
            batch=16,
            device=0,
            workers=4,
            project=None,
            name="train",
            exist_ok=False,
            pretrained=True,
            resume=False,
            optimizer="auto",
            lr0=0.001,
            patience=20,
            save_period=10,
            amp=True,
            freeze=None,
            cache=False,
            plots=True,
            val=True,
            verbose=True,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=5.0,
            translate=0.2,
            scale=0.6,
            shear=2.0,
            perspective=0.001,
            flipud=0.1,
            fliplr=0.5,
            bgr=0.0,
            mosaic=1.0,
            mixup=0.2,
            copy_paste=0.3,
            copy_paste_mode="flip",
            auto_augment="randaugment",
            erasing=0.4,
            crop_fraction=1.0,
            close_mosaic=10,
            class_weights=[2.0, 1.0],
        )
    except ValueError as exc:
        assert "class_weights" in str(exc)
    else:
        raise AssertionError("Expected class_weights to be rejected")


def test_plot_training_results_creates_summary_png(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        " epoch, train/box_loss, val/box_loss, metrics/mAP50(B), metrics/recall(B), lr/pg0\n"
        "1, 1.2, 1.4, 0.20, 0.30, 0.001\n"
        "2, 0.8, 1.0, 0.45, 0.55, 0.0008\n",
        encoding="utf-8",
    )

    output_path = plot_training_results(csv_path)

    assert output_path == tmp_path / "training_summary.png"
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_plot_training_results_from_run_uses_model_trainer_save_dir(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "epoch,train/box_loss,metrics/mAP50(B)\n"
        "1,1.0,0.2\n"
        "2,0.7,0.4\n",
        encoding="utf-8",
    )

    class Trainer:
        save_dir = tmp_path

    class Model:
        trainer = Trainer()

    output_path = plot_training_results_from_run(None, Model())

    assert output_path == tmp_path / "training_summary.png"
    assert output_path.is_file()


def test_add_training_plot_callback_registers_fit_epoch_callback():
    calls = []

    class Model:
        def add_callback(self, event, callback):
            calls.append((event, callback))

    add_training_plot_callback(Model())

    assert len(calls) == 1
    assert calls[0][0] == "on_fit_epoch_end"
