# YOLO26 自定义训练循环重构 — 设计文档

**日期**: 2026-06-16  
**目标**: 废弃 ultralytics.YOLO.train()，重回 yolov8-pytorch 风格的自定义训练循环，所有路径简单化。

## 1. 动机

当前 yolo26 项目有以下问题：

- **路径混乱**：config.py 中硬编码绝对路径 (`C:\Users\EDY\...`)、相对路径、ultralytics 内部路径拼接错误
- **断点续训不可用**：训练保存路径与 resume 查找路径不一致
- **过度依赖 ultralytics**：`YOLO.train()` 是个黑盒，路径/checkpoint/回调行为不可控

参考项目 `yolov8-pytorch` 的风格：所有配置集中在 `train.py` 顶部、简单相对路径、自定义训练循环。

## 2. 设计原则

- **简单路径**：所有路径为相对于项目根目录的字符串，写在 `train.py` 顶部
- **官方代码复用**：模型定义和数据加载来自 ultralytics（不重写 backbone/dataloader）
- **训练循环自主**：Loss、EMA、LR、fit_one_epoch 自己实现
- **单文件配置**：删除 `config.py`，配置归入 `train.py`

## 3. 文件结构

```
yolo26/
├── train.py                     # 所有配置 + 两阶段训练循环 [重写]
├── yolo.py                      # 推理封装 [重写]
├── predict.py                   # 多模式预测入口 [不变]
├── get_map.py                   # mAP 评估 [微调: 去掉 config.py 依赖]
├── summary.py                   # 模型 FLOPs/参数 [不变]
│
├── nets/
│   ├── __init__.py              # [新建]
│   └── yolo_training.py         # Loss, TaskAlignedAssigner, ModelEMA, LR Scheduler [新建]
│
├── utils/
│   ├── __init__.py              # [已有]
│   ├── utils.py                 # cvtColor, get_classes, resize_image 等 [已有, 不动]
│   ├── utils_fit.py             # fit_one_epoch [新建]
│   ├── callbacks.py             # LossHistory, EvalCallback [新建]
│   ├── utils_bbox.py            # DecodeBox, make_anchors, dist2bbox [新建]
│   └── utils_map.py             # mAP 计算 [新建]
│
├── scripts/
│   └── prepare_yolo_dataset.py  # 数据集准备 [已有, 不动]
│
├── model_data/                  # 模型权重 + 字体
├── datasets/                    # 标准 YOLO 格式数据
└── logs/                        # 训练输出
```

### 删除的文件

- `config.py` — 全部删除，配置归入 `train.py`和 `yolo.py`等其他相关文件
- `tests/test_helpers.py` — 如引用了 config，需要适配

## 4. 模块设计

### 4.1 `train.py` — 配置与训练入口

所有训练参数在 `if __name__ == "__main__":` 顶部定义，注释风格与原版 yolov8-pytorch 完全一致：

```python
if __name__ == "__main__":
    #---------------------------------#
    #   Cuda    是否使用Cuda
    #           没有GPU可以设置成False
    #---------------------------------#
    Cuda            = True
    #----------------------------------------------#
    #   Seed    用于固定随机种子
    #           使得每次独立训练都可以获得一样的结果
    #----------------------------------------------#
    seed            = 11
    #---------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #               可减少约一半的显存、需要pytorch1.7.1以上
    #---------------------------------------------------------------------#
    fp16            = False
    #---------------------------------------------------------------------#
    #   classes_path    指向model_data下的txt，与自己训练的数据集相关
    #                   训练前一定要修改classes_path，使其对应自己的数据集
    #---------------------------------------------------------------------#
    classes_path    = 'datasets/datasets.yaml'
    #----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。模型的 预训练权重 对不同数据集是通用的，因为特征是通用的。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #   预训练权重对于99%的情况都必须要用，不用的话主干部分的权值太过随机，特征提取效果不明显，网络训练的结果也不会好
    #
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的 冻结阶段 或者 解冻阶段 的参数，来保证模型epoch的连续性。
    #
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的。
    #   如果想要让模型从0开始训练，则设置model_path = ''，下面的Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #
    #   一般来讲，网络从0开始的训练效果会很差，因为权值太过随机，特征提取效果不明显，因此非常、非常、非常不建议大家从0开始训练！
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = 'model_data/yolo26x.pt'
    #------------------------------------------------------#
    #   input_shape     输入的shape大小，一定要是32的倍数
    #------------------------------------------------------#
    input_shape     = [640, 640]
    #----------------------------------------------------------------------------------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。设置冻结阶段是为了满足机器性能不足的同学的训练需求。
    #   冻结训练需要的显存较小，显卡非常差的情况下，可设置Freeze_Epoch等于UnFreeze_Epoch，Freeze_Train = True，此时仅仅进行冻结训练。
    #
    #   在此提供若干参数设置建议，各位训练者根据自己的需求进行灵活调整：
    #   （一）从整个模型的预训练权重开始训练：
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 1e-3，weight_decay = 0。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-3，weight_decay = 0。（不冻结）
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 300，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 5e-4。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 300，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 5e-4。（不冻结）
    #       其中：UnFreeze_Epoch可以在100-300之间调整。
    #   （二）从0开始训练：
    #       Init_Epoch = 0，UnFreeze_Epoch >= 300，Unfreeze_batch_size >= 16，Freeze_Train = False（不冻结训练）
    #       其中：UnFreeze_Epoch尽量不小于300。optimizer_type = 'sgd'，Init_lr = 1e-2，mosaic = True。
    #   （三）batch_size的设置：
    #       在显卡能够接受的范围内，以大为好。显存不足与数据集大小无关，提示显存不足（OOM或者CUDA out of memory）请调小batch_size。
    #       受到BatchNorm层影响，batch_size最小为2，不能为1。
    #       正常情况下Freeze_batch_size建议为Unfreeze_batch_size的1-2倍。不建议设置的差距过大，因为关系到学习率的自动调整。
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    #   Init_Epoch          模型当前开始的训练世代，其值可以大于Freeze_Epoch，如设置：
    #                       Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100
    #                       会跳过冻结阶段，直接从60代开始，并调整对应的学习率。
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
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    #------------------------------------------------------------------#
    Init_lr             = 1e-3
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    #------------------------------------------------------------------#
    optimizer_type      = "sgd"
    momentum            = 0.937
    weight_decay        = 5e-4
    #------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
    #------------------------------------------------------------------#
    lr_decay_type       = "cos"
    #------------------------------------------------------------------#
    #   mosaic              马赛克数据增强。
    #   mosaic_prob         每个step有多少概率使用mosaic数据增强，默认50%。
    #
    #   mixup               是否使用mixup数据增强，仅在mosaic=True时有效。
    #                       只会对mosaic增强后的图片进行mixup的处理。
    #   mixup_prob          有多少概率在mosaic后使用mixup数据增强，默认50%。
    #                       总的mixup概率为mosaic_prob * mixup_prob。
    #
    #   special_aug_ratio   参考YoloX，由于Mosaic生成的训练图片，远远脱离自然图片的真实分布。
    #                       当mosaic=True时，本代码会在special_aug_ratio范围内开启mosaic。
    #                       默认为前70%个epoch，100个世代会开启70个世代。
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
    #   save_dir        权值与日志文件保存的文件夹
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   eval_flag       是否在训练时进行评估，评估对象为验证集
    #                   评估需要消耗较多的时间，频繁评估会导致训练非常慢
    #------------------------------------------------------------------#
    eval_flag           = True
    #------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   内存较小的电脑可以设置为2或者0
    #------------------------------------------------------------------#
    num_workers         = 4
```

**训练流程**：

1. 加载 ultralytics 模型: `nn.Module = YOLO(model_path).model`
2. 获取类别: `get_classes(classes_path)`
3. 构建 DataLoader（ultralytics API）
4. Phase 1: 冻结 backbone → 训练 (Init_Epoch → Freeze_Epoch)
5. Phase 2: 解冻全部 → 训练 (Freeze_Epoch → UnFreeze_Epoch)
6. 每个 epoch: `fit_one_epoch(train)` + `fit_one_epoch(val)` + LossHistory + EvalCallback + 保存

**断点续训**：

```python
if Init_Epoch > 0:
    model_path = 'logs/xxx/last_epoch_weights.pth'
```

用户手动指定 `model_path` 和 `Init_Epoch`，不搞自动发现。简单明确。

**checkpoint 格式**：

```python
{
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
}
```

保存文件：
- `ep<epoch>-loss<loss>-val_loss<val_loss>.pth` — 定期保存
- `best_epoch_weights.pth` — 最佳 val_loss
- `last_epoch_weights.pth` — 断点续训用

### 4.2 `nets/yolo_training.py` — Loss / EMA / LR Scheduler

#### Loss

针对 yolo26 特点：

- **reg_max=1**：无 DFL loss，只计算 box loss (CIoU) + cls loss (BCE)
- **dual-head**：训练时只使用 `one2many` 分支的输出
- TaskAlignedAssigner：从原版 yolov8-pytorch 搬过来，参数适配 yolo26

```python
class Loss:
    def __init__(self, model):
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=..., alpha=0.5, beta=6.0)
        # 无 DFL: reg_max=1

    def __call__(self, preds, batch):
        # preds: 模型前向输出 (one2many branch)
        # 计算 box loss (CIoU) + cls loss (BCE)
        # 返回 loss.sum()
```

#### ModelEMA

从原版搬过来，标准指数滑动平均。

#### LR Scheduler

支持 warmup + cos 退火 / step 衰减。

### 4.3 `utils/utils_fit.py` — fit_one_epoch

与原版 yolov8-pytorch 一致：

```python
def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history,
                  eval_callback, optimizer, epoch, epoch_step,
                  epoch_step_val, gen, gen_val, Epoch, cuda,
                  fp16, scaler, save_period, save_dir, local_rank=0):
    # 训练循环
    for iteration, batch in enumerate(gen):
        images, bboxes = batch
        outputs = model_train(images)
        loss_value = yolo_loss(outputs, bboxes)
        loss_value.backward()
        optimizer.step()
        if ema: ema.update(model_train)

    # 验证循环
    for iteration, batch in enumerate(gen_val):
        ...

    # 保存 checkpoint
    if ema:
        torch.save(ema.ema.state_dict(), ...)
    else:
        torch.save(model.state_dict(), ...)
```

### 4.4 `utils/callbacks.py` — LossHistory / EvalCallback

LossHistory：TensorBoard 写入、loss 曲线图  
EvalCallback：定期在验证集上计算 mAP

### 4.5 `utils/utils_bbox.py` — 边界框工具

`make_anchors`, `dist2bbox`, `DecodeBox`（用于推理时的解码和 NMS）。

### 4.6 `utils/utils_map.py` — mAP 计算

从原版搬过来，get_map / get_coco_map。

### 4.7 `yolo.py` — 推理

重写，不再使用 `ultralytics.YOLO`。直接加载自己训练的 `.pth` 权重，手动前向传播 + 解码 + NMS + 渲染。

保留接口：`detect_image`, `get_FPS`, `detect_heatmap`, `convert_to_onnx`

### 4.8 `get_map.py` — 微调

去掉 `from config import ...`，改为直接在文件里配置路径。

## 5. 数据流

```
datasets/datasets.yaml
    ↓
ultralytics.data.build_dataloader()   ← YOLO 格式，官方加载
    ↓
train.py: for epoch in range(...):
    fit_one_epoch(train_loader, val_loader)
        ↓
    model(images) → outputs (one2many)
        ↓
    Loss(outputs, bboxes) → box_loss + cls_loss
        ↓
    optimizer.step() / ema.update()
        ↓
    loss_history / eval_callback
        ↓
    torch.save(checkpoint, logs/xxx/)
```

## 6. 风险与注意事项

1. **ultralytics 版本兼容**：模型定义 `YOLO(...).model` 和 DataLoader API 依赖 ultralytics 内部接口，升级可能破坏
2. **dual-head 适配**：确认 `model(images)` 训练模式返回的是 one2many 分支输出，不是 one2one
3. **ckpt 兼容转换**：需要将 ultralytics 预训练权重转为自己的 checkpoint 格式，保持 key 名称一致
4. **get_map.py 路径**：需要同步修改，去掉 config.py 依赖

## 7. 验收标准

- [ ] `config.py` 已删除
- [ ] `train.py` 顶部所有路径为简单相对字符串
- [ ] 自定义训练循环可以启动并完成一个 epoch（Loss 正常下降）
- [ ] 两阶段训练可以正常切换（冻结 → 解冻）
- [ ] checkpoint 可以正常保存和续训
- [ ] `yolo.py` 推理可用（加载自己训练的权重）
- [ ] `predict.py` 所有模式可用
- [ ] `get_map.py` 可用
- [ ] 所有中文注释风格与原版保持一致
