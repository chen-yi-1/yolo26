from train import build_train_kwargs


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
