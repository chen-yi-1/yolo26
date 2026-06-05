#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import datetime
import os

from ultralytics import YOLO
from ultralytics.utils.plotting import plot_results


def _add_per_epoch_plotting(model):
    """在每个epoch结束后更新results.png，即使训练中断也能看到曲线。"""
    def on_fit_epoch_end(trainer):
        if trainer.csv.exists():
            plot_results(file=trainer.csv)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


def _add_phase_checkpoint(model, ckpt_name='phase_last.pt'):
    """在训练过程中保存一份带 optimizer 状态的 checkpoint 副本。

    ultralytics 在训练正常结束后会自动剥离 last.pt 中的 optimizer/epoch 状态，
    导致 Phase 2 无法 resume。此回调在最后一个 epoch 结束时额外保存一份未剥离的副本。
    """
    import shutil

    def on_fit_epoch_end(trainer):
        shutil.copy(trainer.last, trainer.save_dir / 'weights' / ckpt_name)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


def _train_with_resume(model, **train_kwargs):
    """调用 model.train()，并绕过 PyTorch 2.6+ 的 weights_only=True 默认值。

    PyTorch 2.6+ 把 torch.load 默认改为 weights_only=True，而 ultralytics 内部
    check_resume 调用 torch.load 时未显式传 weights_only=False，导致无法读取
    checkpoint 中的 optimizer state dict，从而误报 "not a resumable training checkpoint"。
    """
    import torch

    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False})
    try:
        return model.train(**train_kwargs)
    finally:
        torch.load = _orig_load

'''
训练自己的目标检测模型一定需要注意以下几点：
1、训练前仔细检查自己的格式是否满足要求，该库要求数据集格式为YOLO格式，需要准备好的内容有输入图片和标签
   输入图片为.jpg图片，无需固定大小，传入训练前会自动进行resize。
   灰度图会自动转成RGB图片进行训练，无需自己修改。
   输入图片如果后缀非jpg，需要自己批量转成jpg后再开始训练。

   标签为.txt格式，每张图片对应一个同名txt，文件中会有需要检测的目标信息。

2、损失值的大小用于判断是否收敛，比较重要的是有收敛的趋势，即验证集损失不断下降，如果验证集损失基本上不改变的话，模型基本上就收敛了。
   损失值的具体大小并没有什么意义，大和小只在于损失的计算方式，并不是接近于0才好。如果想要让损失好看点，可以直接到对应的损失函数里面除上10000。
   训练过程中的损失值会保存在 runs/detect/logs/ 下

3、训练好的权值文件保存在 runs/detect/logs/ 中，每个训练世代（Epoch）包含若干训练步长（Step），每个训练步长（Step）进行一次梯度下降。
   如果只是训练了几个Step是不会保存的，Epoch和Step的概念要捋清楚一下。
'''
if __name__ == "__main__":
    #---------------------------------#
    #   Cuda    是否使用Cuda
    #           没有GPU可以设置成False
    #---------------------------------#
    Cuda            = False
    #----------------------------------------------#
    #   Seed    用于固定随机种子
    #           使得每次独立训练都可以获得一样的结果
    #----------------------------------------------#
    seed            = 11
    #---------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #               可减少约一半的显存、需要pytorch1.7.1以上
    #---------------------------------------------------------------------#
    fp16            = True
    #----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。模型的 预训练权重 对不同数据集是通用的，因为特征是通用的。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #   预训练权重对于99%的情况都必须要用，不用的话主干部分的权值太过随机，特征提取效果不明显，网络训练的结果也不会好
    #
    #   如果训练过程中存在中断训练的操作，可以在断点续训时将model_path设置成runs/detect/logs/下的last.pt权值文件。
    #
    #   YOLO26 预训练权重路径（支持自动下载：yolo26n.pt / yolo26s.pt / yolo26m.pt / yolo26l.pt / yolo26x.pt）
    #   如果想要让模型从0开始训练，则设置model_path = 'yolo26x.yaml'，下面的Freeze_Train = False，此时从零开始训练，且没有冻结主干的过程。
    #
    #   一般来讲，网络从0开始的训练效果会很差，因为权值太过随机，特征提取效果不明显，因此非常、非常、非常不建议大家从0开始训练！
    #   从0开始训练有两个方案：
    #   1、得益于Mosaic数据增强方法强大的数据增强能力，将UnFreeze_Epoch设置的较大（300及以上）、batch较大（16及以上）、数据较多（万以上）的情况下，
    #      可以设置mosaic=True，直接随机初始化参数开始训练，但得到的效果仍然不如有预训练的情况。（像COCO这样的大数据集可以这样做）
    #   2、了解imagenet数据集，首先训练分类模型，获得网络的主干部分权值，分类模型的 主干部分 和该模型通用，基于此进行训练。
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = 'model_data/yolo26n.pt'
    #---------------------------------------------------------------------#
    #   data_yaml        YOLO格式的数据集配置文件路径
    #                    文件中应包含 train/val 路径 和 names 类别名
    #---------------------------------------------------------------------#
    data_yaml       = 'dataset.yaml'
    #------------------------------------------------------#
    #   input_shape     输入的shape大小，一定要是32的倍数
    #------------------------------------------------------#
    input_shape     = [640, 640]
    #----------------------------------------------------------------------------------------------------------------------------#
    #   YOLO26 训练策略（ultralytics optimizer="auto" 默认使用 Adam）：
    #   小数据集 + 预训练模型 → Adam 优化器，100 epochs 即可收敛
    #
    #   参数建议：
    #   （一）加载预训练权重（推荐）：
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 1e-3，weight_decay = 0。
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-3，weight_decay = 0。
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 200，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 5e-4。
    #   （二）从零开始训练（不推荐）：
    #       UnFreeze_Epoch >= 300，Unfreeze_batch_size >= 16，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2，mosaic = True。
    #   （三）batch_size：
    #       显存不足请调小batch_size。受BatchNorm影响，batch_size最小为2。
    #       正常情况下Freeze_batch_size建议为Unfreeze_batch_size的1-2倍。
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    #   Init_Epoch          模型当前开始的训练世代，其值可以大于Freeze_Epoch，如设置：
    #                       Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100
    #                       会跳过冻结阶段，直接从60代开始。
    #                       （断点续练时使用）
    #   Freeze_Epoch        模型冻结训练的Freeze_Epoch
    #                       (当Freeze_Train=False时失效)
    #   Freeze_batch_size   模型冻结训练的batch_size
    #                       (当Freeze_Train=False时失效)
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 32
    #------------------------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    #   UnFreeze_Epoch          模型总共训练的epoch
    #                           YOLO26 小数据集推荐 100 epochs
    #   Unfreeze_batch_size     模型在解冻后的batch_size
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 16
    #------------------------------------------------------------------#
    #   Freeze_Train    是否进行冻结训练
    #                   默认先冻结主干训练后解冻训练。
    #------------------------------------------------------------------#
    Freeze_Train        = True

    #------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #                   Adam: 1e-3    SGD: 1e-2
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    #------------------------------------------------------------------#
    Init_lr             = 1e-3
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有auto、adam、sgd
    #                   auto: ultralytics默认，YOLO26自动使用Adam
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #                   当使用Adam时作为beta1
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam建议设置为0。
    #------------------------------------------------------------------#
    optimizer_type      = "auto"
    momentum            = 0.937
    weight_decay        = 0
    #------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
    #------------------------------------------------------------------#
    lr_decay_type       = "cos"
    #------------------------------------------------------------------#
    #   mosaic              马赛克数据增强。
    #   mosaic_prob         每个step有多少概率使用mosaic数据增强，默认100%。
    #
    #   mixup               是否使用mixup数据增强，仅在mosaic=True时有效。
    #                       只会对mosaic增强后的图片进行mixup的处理。
    #   mixup_prob          有多少概率在mosaic后使用mixup数据增强，默认50%。
    #                       总的mixup概率为mosaic_prob * mixup_prob。
    #
    #   special_aug_ratio   参考YoloX，由于Mosaic生成的训练图片，远远脱离自然图片的真实分布。
    #                       当mosaic=True时，本代码会在special_aug_ratio范围内开启mosaic。
    #                       默认为前70%个epoch，100个世代会开启70个世代。
    #                       对应ultralytics的close_mosaic参数：最后N个epoch关闭mosaic
    #------------------------------------------------------------------#
    mosaic              = True
    mosaic_prob         = 0.5
    mixup               = True
    mixup_prob          = 0.5
    special_aug_ratio   = 0.7
    #------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值
    #------------------------------------------------------------------#
    save_period         = 10
    #------------------------------------------------------------------#
    #   save_dir        训练输出的 project 名称（ultralytics 实际路径为 runs/detect/{save_dir}/{train_name}/）
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   eval_flag       是否在训练时进行评估，评估对象为验证集
    #                   安装pycocotools库后，评估体验更佳。
    #   注意：ultralytics 默认每个epoch都验证一次，不支持eval_period间隔设置
    #   如需减少验证频率，需要使用回调函数自定义验证逻辑
    #   此处获得的mAP会与get_map.py获得的会有所不同，原因有二：
    #   （一）此处获得的mAP为验证集的mAP。
    #   （二）此处设置评估参数较为保守，目的是加快评估速度。
    #------------------------------------------------------------------#
    eval_flag           = True
    #------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   内存较小的电脑可以设置为2或者0
    #------------------------------------------------------------------#
    num_workers         = 4

    #------------------------------------------------------#
    #   设置用到的显卡
    #------------------------------------------------------#
    device = 'cuda' if Cuda else 'cpu'

    #------------------------------------------------------#
    #   生成训练日志名称（Phase 1 和 Phase 2 共用）
    #------------------------------------------------------#
    train_name = f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    #------------------------------------------------------#
    #   断点续训：Init_Epoch > 0 时自动发现最新 checkpoint
    #------------------------------------------------------#
    if Init_Epoch > 0:
        base_dir = os.path.join('runs', 'detect', save_dir)
        checkpoints = []
        if os.path.isdir(base_dir):
            checkpoints = sorted(
                [os.path.join(base_dir, d, 'weights', 'last.pt')
                 for d in os.listdir(base_dir)
                 if os.path.isfile(os.path.join(base_dir, d, 'weights', 'last.pt'))],
                key=os.path.getmtime,
            )
        if checkpoints:
            model_path = checkpoints[-1]
            # 从 checkpoint 路径反推 train_name，续训结果写回原目录
            # 路径格式: runs/detect/{save_dir}/{train_name}/weights/last.pt
            resume_train_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
            # 验证 checkpoint 是否包含可续训状态（epoch + optimizer）
            import torch
            ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
            ckpt_epoch = ckpt.get('epoch', None)
            if ckpt_epoch is None:
                print(f"\n[Error] Checkpoint missing epoch/optimizer state, cannot resume!")
                print(f"  {model_path}")
                print(f"  This is unexpected for a last.pt saved during training. Possible causes:")
                print(f"  1. Training was killed before completing the first epoch")
                print(f"  2. The file was corrupted or overwritten")
                print(f"  3. PyTorch version mismatch when loading the checkpoint")
                raise RuntimeError("Resume failed: checkpoint has no training state. "
                                   "Set Init_Epoch=0 or fix the checkpoint path.")
            actual_init = ckpt_epoch + 1  # ckpt epoch is 0-indexed → Init_Epoch is 1-indexed
            if actual_init != Init_Epoch:
                print(f"\n[Warn] Init_Epoch={Init_Epoch} but checkpoint last completed epoch={actual_init}.")
                print(f"       Phase/skip logic will use Init_Epoch={Init_Epoch}, but training will")
                print(f"       resume from epoch {actual_init+1} per the checkpoint state.")
            print(f"\n[Resume] Init_Epoch={Init_Epoch}, checkpoint epoch={ckpt_epoch+1}")
            print(f"  {model_path}")
            print(f"  Results will be saved to: runs/detect/{save_dir}/{resume_train_name}/")
            train_name = resume_train_name
        else:
            print(f"\n[Error] Init_Epoch={Init_Epoch} but no checkpoint found under {base_dir}.")
            print(f"  Set Init_Epoch=0 to start fresh training, or ensure a training run exists.")
            raise RuntimeError("Resume failed: no checkpoint found. "
                               "Set Init_Epoch=0 or check save_dir.")

    from utils.utils import show_config
    show_config(
        model_path = model_path, data_yaml = data_yaml, input_shape = input_shape,
        Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch,
        Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size,
        Freeze_Train = Freeze_Train,
        Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type,
        momentum = momentum, lr_decay_type = lr_decay_type,
        save_period = save_period, save_dir = save_dir, num_workers = num_workers,
    )

    if Freeze_Train:
        if Init_Epoch >= Freeze_Epoch:
            print(f"\n[Info] Init_Epoch={Init_Epoch} >= Freeze_Epoch={Freeze_Epoch}, skipping freeze phase.")
        if Init_Epoch >= UnFreeze_Epoch:
            raise ValueError("Init_Epoch must be less than UnFreeze_Epoch!")

    #----------------------------------------------#
    #   总训练世代指的是遍历全部数据的总次数
    #   总训练步长指的是梯度下降的总次数
    #   每个训练世代包含若干训练步长，每个训练步长进行一次梯度下降。
    #   此处仅建议最低训练世代，上不封顶，计算时只考虑了解冻部分
    #----------------------------------------------#
    wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
    # 粗略估计训练步长，实际取决于数据集大小
    # TODO: 如需精确估计，可从 data_yaml 中读取数据集实际图片数量
    estimated_images = 1000
    if os.path.exists(data_yaml):
        try:
            import yaml
            with open(data_yaml, encoding='utf-8') as f:
                ydata = yaml.safe_load(f)
            train_dir = os.path.join(ydata.get('path', ''), ydata.get('train', ''))
            if os.path.isdir(train_dir):
                import glob
                estimated_images = len(glob.glob(os.path.join(train_dir, '*.[jJ][pP][gG]'))) or estimated_images
        except Exception:
            pass
    total_step  = estimated_images // Unfreeze_batch_size * UnFreeze_Epoch
    if total_step <= wanted_step:
        print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m"%(optimizer_type, wanted_step))
        print("\033[1;33;44m[Warning] 如果数据集较小，请增加UnFreeze_Epoch以满足足够的训练步长。\033[0m")

    # 两个阶段的公共训练参数（修改一处即可同步）
    train_args = dict(
        data=data_yaml,
        imgsz=input_shape[0],
        device=device,
        workers=num_workers,
        optimizer=optimizer_type,
        lr0=Init_lr,
        lrf=Min_lr / Init_lr,
        momentum=momentum,
        weight_decay=weight_decay,
        cos_lr=(lr_decay_type == "cos"),
        mosaic=mosaic_prob if mosaic else 0.0,
        mixup=mixup_prob if (mosaic and mixup) else 0.0,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        amp=fp16,
        seed=seed,
        project=save_dir,
        name=train_name,
        exist_ok=True,              # 允许续训时覆盖已有目录
        save_period=save_period,
        val=eval_flag,
        plots=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
    )

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   UnFreeze_Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    # ================================ #
    #   Phase 1: Freeze Backbone
    # ================================ #
    # 判断是否从断点续训：Init_Epoch > 0 表示 model_path 指向的是一个训练保存的 checkpoint
    is_resuming = Init_Epoch > 0
    freeze_phase_epochs = Freeze_Epoch - Init_Epoch
    freeze_save_dir = None  # 记录 Phase 1 的实际保存路径，供 Phase 2 使用
    if Freeze_Train and freeze_phase_epochs > 0:
        print(f"\n[Phase 1] Freezing backbone for {freeze_phase_epochs} epochs "
              f"(epoch {Init_Epoch}→{Freeze_Epoch}, batch={Freeze_batch_size})"
              f"{' [resume]' if is_resuming else ''}")
        model = YOLO(model_path)
        _add_per_epoch_plotting(model)
        _add_phase_checkpoint(model)    # 保存未剥离的 checkpoint 供 Phase 2 resume
        _train_with_resume(model,
            **train_args,
            epochs=Freeze_Epoch,
            batch=Freeze_batch_size,
            warmup_epochs=0 if is_resuming else 3.0,
            close_mosaic=0,             # 冻结阶段不关闭mosaic
            freeze=10,                  # 冻结前10层（backbone）
            resume=is_resuming,
        )
        freeze_save_dir = str(model.trainer.save_dir)

    # ================================ #
    #   Phase 2: Unfreeze All
    # ================================ #
    remaining = UnFreeze_Epoch - max(Init_Epoch, Freeze_Epoch if Freeze_Train else Init_Epoch)
    if remaining > 0:
        start_epoch = max(Init_Epoch, Freeze_Epoch if Freeze_Train else Init_Epoch)
        print(f"\n[Phase 2] Unfreezing all layers for {remaining} epochs "
              f"(epoch {start_epoch}→{UnFreeze_Epoch}, batch={Unfreeze_batch_size})"
              f"{' [resume]' if (freeze_save_dir is not None or is_resuming) else ''}")
        # Phase 1 运行过则从 last.pt 继续；否则从用户指定的 model_path
        if freeze_save_dir is not None:
            last_pt = os.path.join(freeze_save_dir, 'weights', 'phase_last.pt')
            resume_phase2 = True   # Phase 1 刚跑完，last.pt 一定是训练断点
        else:
            last_pt = model_path
            resume_phase2 = is_resuming   # 只有用户传入的是训练断点时才 resume

        close_mosaic_unfreeze = int(UnFreeze_Epoch * (1.0 - special_aug_ratio)) if mosaic else 0
        model = YOLO(last_pt)
        _add_per_epoch_plotting(model)
        _train_with_resume(model,
            **train_args,
            epochs=UnFreeze_Epoch,
            batch=Unfreeze_batch_size,
            warmup_epochs=0 if resume_phase2 else 3.0,
            close_mosaic=close_mosaic_unfreeze,
            freeze=None,                # 解冻所有层
            resume=resume_phase2,
        )
        final_save_dir = str(model.trainer.save_dir)
    elif freeze_save_dir is not None:
        final_save_dir = freeze_save_dir
    else:
        final_save_dir = None

    if final_save_dir:
        print(f"\nTraining complete. Results saved in {final_save_dir}")
    else:
        print(f"\nTraining complete.")
