#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import datetime
import os
import shutil
from contextlib import contextmanager

import torch

from config import DATA, TASK, TRAIN, get_model_path, get_run_task
from scripts.prepare_yolo_dataset import prepare_yolo_dataset


@contextmanager
def torch_load_weights_only_false():
    """Temporarily force torch.load(weights_only=False) for ultralytics resume."""
    torch_load = torch.load
    torch.load = lambda *a, **kw: torch_load(*a, **{**kw, 'weights_only': False})
    try:
        yield
    finally:
        torch.load = torch_load


def phase_train_names(train_name, is_resuming):
    def base_name(name):
        for suffix in ("_freeze", "_unfreeze"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    if is_resuming:
        return train_name, f"{base_name(train_name)}_unfreeze"
    return f"{train_name}_freeze", f"{train_name}_unfreeze"


def phase2_epochs(init_epoch, freeze_epoch, unfreeze_epoch, freeze_train):
    start_epoch = max(init_epoch, freeze_epoch if freeze_train else init_epoch)
    return unfreeze_epoch - start_epoch


def _setup_callbacks(model, save_phase_ckpt=False):
    """统一注册训练回调：per-epoch 绘图 + Phase 1 保存未剥离的 checkpoint。

    ultralytics 在训练正常结束后会自动剥离 last.pt 的 optimizer/epoch 状态，
    导致 Phase 2 无法 resume。save_phase_ckpt=True 时额外保存 phase_last.pt。

    注意：phase_last.pt 的复制必须挂在 on_model_save 上，不能挂在 on_fit_epoch_end。
    final_eval() 会在 strip_optimizer 之后再次触发 on_fit_epoch_end，导致
    phase_last.pt 被覆盖为已剥离的版本。
    """
    from ultralytics.utils.plotting import plot_results

    def on_fit_epoch_end(trainer):
        if trainer.csv.exists():
            plot_results(file=trainer.csv)

    def on_model_save(trainer):
        if save_phase_ckpt:
            shutil.copy(trainer.last, trainer.save_dir / 'weights' / 'phase_last.pt')

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    if save_phase_ckpt:
        model.add_callback("on_model_save", on_model_save)


'''
训练自己的目标检测模型一定需要注意以下几点：
1、训练前仔细检查自己的格式是否满足要求，该库要求数据集格式为YOLO格式，需要准备好的内容有输入图片和标签
   输入图片为.jpg图片，无需固定大小，传入训练前会自动进行resize。
   灰度图会自动转成RGB图片进行训练，无需自己修改。
   输入图片如果后缀非jpg，需要自己批量转成jpg后再开始训练。

   标签为.txt格式，每张图片对应一个同名txt，文件中会有需要检测的目标信息。

2、损失值的大小用于判断是否收敛，比较重要的是有收敛的趋势，即验证集损失不断下降，如果验证集损失基本上不改变的话，模型基本上就收敛了。
   损失值的具体大小并没有什么意义，大和小只在于损失的计算方式，并不是接近于0才好。如果想要让损失好看点，可以直接到对应的损失函数里面除上10000。
   训练过程中的损失值会保存在 runs/segment/logs/ 下

3、训练好的权值文件保存在 runs/segment/logs/ 中，每个训练世代（Epoch）包含若干训练步长（Step），每个训练步长（Step）进行一次梯度下降。
   如果只是训练了几个Step是不会保存的，Epoch和Step的概念要捋清楚一下。
'''
if __name__ == "__main__":
    from ultralytics import YOLO

    #---------------------------------#
    #   Cuda    是否使用Cuda
    #           没有GPU可以设置成False
    #---------------------------------#
    Cuda            = TRAIN["cuda"]
    #----------------------------------------------#
    #   Seed    用于固定随机种子
    #           使得每次独立训练都可以获得一样的结果
    #----------------------------------------------#
    seed            = TRAIN["seed"]
    #---------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #               可减少约一半的显存、需要pytorch1.7.1以上
    #---------------------------------------------------------------------#
    fp16            = TRAIN["fp16"]
    #----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。模型的 预训练权重 对不同数据集是通用的，因为特征是通用的。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #   预训练权重对于99%的情况都必须要用，不用的话主干部分的权值太过随机，特征提取效果不明显，网络训练的结果也不会好
    #
    #   如果训练过程中存在中断训练的操作，可以在断点续训时将model_path设置成runs/segment/logs/下的last.pt权值文件。
    #
    #   YOLO26 Segment 预训练权重路径（支持自动下载：yolo26n-seg.pt / yolo26s-seg.pt / yolo26m-seg.pt / yolo26l-seg.pt / yolo26x-seg.pt）
    #   如果想要让模型从0开始训练，则设置model_path = 'yolo26x.yaml'，下面的Freeze_Train = False，此时从零开始训练，且没有冻结主干的过程。
    #
    #   一般来讲，网络从0开始的训练效果会很差，因为权值太过随机，特征提取效果不明显，因此非常、非常、非常不建议大家从0开始训练！
    #   从0开始训练有两个方案：
    #   1、得益于Mosaic数据增强方法强大的数据增强能力，将UnFreeze_Epoch设置的较大（300及以上）、batch较大（16及以上）、数据较多（万以上）的情况下，
    #      可以设置mosaic=True，直接随机初始化参数开始训练，但得到的效果仍然不如有预训练的情况。（像COCO这样的大数据集可以这样做）
    #   2、了解imagenet数据集，首先训练分类模型，获得网络的主干部分权值，分类模型的 主干部分 和该模型通用，基于此进行训练。
    #----------------------------------------------------------------------------------------------------------------------------#
    task            = TASK
    model_path      = get_model_path(task)
    #---------------------------------------------------------------------#
    #   data_yaml        YOLO格式的数据集配置文件路径
    #                    文件中应包含 train/val 路径 和 names 类别名
    #---------------------------------------------------------------------#
    data_yaml       = DATA["yaml"]
    #------------------------------------------------------#
    #   input_shape     输入的shape大小，一定要是32的倍数
    #------------------------------------------------------#
    input_shape     = TRAIN["input_shape"]
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
    Init_Epoch          = TRAIN["init_epoch"]
    Freeze_Epoch        = TRAIN["freeze_epoch"]
    Freeze_batch_size   = TRAIN["freeze_batch_size"]
    #------------------------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    #   UnFreeze_Epoch          模型总共训练的epoch
    #                           YOLO26 小数据集推荐 100 epochs
    #   Unfreeze_batch_size     模型在解冻后的batch_size
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = TRAIN["unfreeze_epoch"]
    Unfreeze_batch_size = TRAIN["unfreeze_batch_size"]
    #------------------------------------------------------------------#
    #   Freeze_Train    是否进行冻结训练
    #                   默认先冻结主干训练后解冻训练。
    #------------------------------------------------------------------#
    Freeze_Train        = TRAIN["freeze_train"]

    #------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #                   Adam: 1e-3    SGD: 1e-2
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    #------------------------------------------------------------------#
    Init_lr             = TRAIN["init_lr"]
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
    optimizer_type      = TRAIN["optimizer_type"]
    momentum            = TRAIN["momentum"]
    weight_decay        = TRAIN["weight_decay"]
    #------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
    #------------------------------------------------------------------#
    lr_decay_type       = TRAIN["lr_decay_type"]
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
    mosaic              = TRAIN["mosaic"]
    mosaic_prob         = TRAIN["mosaic_prob"]
    mixup               = TRAIN["mixup"]
    mixup_prob          = TRAIN["mixup_prob"]
    special_aug_ratio   = TRAIN["special_aug_ratio"]
    #------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值
    #------------------------------------------------------------------#
    save_period         = TRAIN["save_period"]
    #------------------------------------------------------------------#
    #   save_dir        训练输出的 project 名称（ultralytics 实际路径为 runs/segment/{save_dir}/{train_name}/）
    #------------------------------------------------------------------#
    save_dir            = TRAIN["save_dir"]
    #------------------------------------------------------------------#
    #   eval_flag       是否在训练时进行评估，评估对象为 dataset.yaml 的验证集
    #   注意：ultralytics 默认每个epoch都验证一次，不支持eval_period间隔设置
    #   如需减少验证频率，需要使用回调函数自定义验证逻辑
    #   get_map.py 使用同一个官方验证入口，可通过 split 选择 val/test。
    #------------------------------------------------------------------#
    eval_flag           = TRAIN["eval_flag"]
    #------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   内存较小的电脑可以设置为2或者0
    #------------------------------------------------------------------#
    num_workers         = TRAIN["num_workers"]

    #------------------------------------------------------#
    #   设置用到的显卡
    #------------------------------------------------------#
    device = 'cuda' if Cuda else 'cpu'
    run_task = get_run_task(task)

    #------------------------------------------------------#
    #   生成训练日志名称（Phase 1 和 Phase 2 共用）
    #------------------------------------------------------#
    train_name = f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    #------------------------------------------------------#
    #   断点续训：Init_Epoch > 0 时自动发现最新 checkpoint
    #------------------------------------------------------#
    if Init_Epoch > 0:
        base_dir = os.path.join('runs', run_task, save_dir)
        ckpt_paths = []
        if os.path.isdir(base_dir):
            ckpt_paths = sorted(
                [os.path.join(base_dir, d, 'weights', 'last.pt')
                 for d in os.listdir(base_dir)
                 if os.path.isfile(os.path.join(base_dir, d, 'weights', 'last.pt'))],
                key=os.path.getmtime,
            )
        if not ckpt_paths:
            raise RuntimeError(
                f"Init_Epoch={Init_Epoch} but no checkpoint found under {base_dir}. "
                f"Set Init_Epoch=0 to start fresh training."
            )

        model_path = ckpt_paths[-1]
        with torch_load_weights_only_false():
            ckpt = torch.load(model_path, map_location='cpu')
        # checkpoint 中 project/resume 字段可能含完整路径导致目录重复，统一修正
        if 'train_args' in ckpt:
            args = ckpt['train_args']
            changed = False
            if args.get('project') != save_dir:
                args['project'] = save_dir
                changed = True
            if isinstance(args.get('resume'), str):
                args['resume'] = True
                changed = True
            if changed:
                torch.save(ckpt, model_path)
        ckpt_epoch = ckpt.get('epoch', None)
        if ckpt_epoch is None:
            raise RuntimeError(
                f"Checkpoint {model_path} has no training state (missing epoch/optimizer).\n"
                f"  This can happen if the training completed normally — ultralytics strips\n"
                f"  optimizer state from last.pt after training finishes. Use phase_last.pt\n"
                f"  instead, or set Init_Epoch=0 to start fresh."
            )

        actual_init = ckpt_epoch + 1  # 0-indexed → 1-indexed
        if actual_init != Init_Epoch:
            print(f"\n[Warn] Init_Epoch={Init_Epoch} but checkpoint last completed epoch={actual_init}.")
            print(f"       Phase logic uses Init_Epoch={Init_Epoch}, but training will resume")
            print(f"       from epoch {actual_init+1} per checkpoint state.")

        # 从 checkpoint 路径反推 train_name，续训结果写回原目录
        # 确保 train_name 只包含目录名，不包含路径前缀
        train_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
        # 防止 train_name 包含路径分隔符导致路径重复
        train_name = os.path.basename(train_name) if os.sep in train_name else train_name
        print(f"\n[Resume] Init_Epoch={Init_Epoch}, checkpoint epoch={ckpt_epoch+1}")
        print(f"  {model_path}")
        print(f"  Results will be saved to: runs/{run_task}/{save_dir}/{train_name}/")

    freeze_train_name, unfreeze_train_name = phase_train_names(train_name, Init_Epoch > 0)

    from utils.utils import show_config
    show_config(
        task = task, model_path = model_path, data_yaml = data_yaml, input_shape = input_shape,
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

    # 两个阶段的公共训练参数
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
        exist_ok=True,
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
    #   Phase 1: Freeze Backbone
    #   Phase 2: Unfreeze All
    #
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代, Freeze_Epoch为冻结世代, UnFreeze_Epoch总世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    is_resuming = Init_Epoch > 0
    freeze_phase_epochs = Freeze_Epoch - Init_Epoch
    freeze_save_dir = None

    if Freeze_Train and freeze_phase_epochs > 0:
        print(f"\n[Phase 1] Freezing backbone for {freeze_phase_epochs} epochs "
              f"(epoch {Init_Epoch}→{Freeze_Epoch}, batch={Freeze_batch_size})"
              f"{' [resume]' if is_resuming else ''}")
        model = YOLO(model_path)
        _setup_callbacks(model, save_phase_ckpt=True)
        phase1_args = dict(
            **train_args,
            name=freeze_train_name,
            epochs=Freeze_Epoch,
            batch=Freeze_batch_size,
            warmup_epochs=0 if is_resuming else 3.0,
            close_mosaic=0,
            freeze=10,
            resume=is_resuming,
        )
        if is_resuming:
            with torch_load_weights_only_false():
                model.train(**phase1_args)
        else:
            model.train(**phase1_args)
        freeze_save_dir = str(model.trainer.save_dir)

    # ================================ #
    #   Phase 2: Unfreeze All
    # ================================ #
    remaining = phase2_epochs(Init_Epoch, Freeze_Epoch, UnFreeze_Epoch, Freeze_Train)
    if remaining > 0:
        start_epoch = max(Init_Epoch, Freeze_Epoch if Freeze_Train else Init_Epoch)

        if freeze_save_dir is not None:
            # Phase 1 刚跑完，加载 phase_last.pt 的模型权重但不 resume。
            # Phase 2 有不同的 epochs/freeze/batch，是新的训练会话。
            last_pt = os.path.join(freeze_save_dir, 'weights', 'phase_last.pt')
            resume_phase2 = False
            tag = ' [from phase_last.pt]'
        else:
            last_pt = model_path
            resume_phase2 = is_resuming
            tag = ' [resume]' if is_resuming else ''

        print(f"\n[Phase 2] Unfreezing all layers for {remaining} epochs "
              f"(epoch {start_epoch}→{UnFreeze_Epoch}, batch={Unfreeze_batch_size}){tag}")

        close_mosaic_unfreeze = int(remaining * (1.0 - special_aug_ratio)) if mosaic else 0
        model = YOLO(last_pt)
        _setup_callbacks(model, save_phase_ckpt=False)
        phase2_args = dict(
            **train_args,
            name=unfreeze_train_name,
            epochs=remaining,
            batch=Unfreeze_batch_size,
            warmup_epochs=0 if resume_phase2 else 3.0,
            close_mosaic=close_mosaic_unfreeze,
            freeze=None,
            resume=resume_phase2,
        )
        if resume_phase2:
            with torch_load_weights_only_false():
                model.train(**phase2_args)
        else:
            model.train(**phase2_args)
        final_save_dir = str(model.trainer.save_dir)
    elif freeze_save_dir is not None:
        final_save_dir = freeze_save_dir
    else:
        final_save_dir = None

    if final_save_dir:
        print(f"\nTraining complete. Results saved in {final_save_dir}")
    else:
        print(f"\nTraining complete.")
