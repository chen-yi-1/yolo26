import os

import torch
from tqdm import tqdm

from utils.utils import get_lr


def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history,
                  eval_callback, optimizer, epoch, epoch_step,
                  epoch_step_val, gen, gen_val, Epoch, cuda,
                  fp16, scaler, save_period, save_dir, local_rank=0):
    """训练和验证一个 epoch, 并保存 checkpoint。

    参考 yolov8-pytorch/utils/utils_fit.py。
    """
    loss = 0
    val_loss = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}',
                    postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        # ultralytics 数据集返回 uint8 (0-255)，模型需要 float32 [0,1]
        images = batch['img'].float() / 255.0
        # 拼接 ultralytics batch dict 为 Loss 期望的格式: [N, 6] = [batch_idx, cls, x, y, w, h]
        batch_targets = torch.cat([
            batch['batch_idx'].float().view(-1, 1),
            batch['cls'].float().view(-1, 1),
            batch['bboxes'].float(),
        ], dim=-1)
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                batch_targets = batch_targets.cuda(local_rank)

        optimizer.zero_grad()

        if not fp16:
            outputs = model_train(images)
            loss_value = yolo_loss(outputs, batch_targets)
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)
            optimizer.step()
        else:
            from torch.amp import autocast
            with autocast('cuda'):
                outputs = model_train(images)
                loss_value = yolo_loss(outputs, batch_targets)
            scaler.scale(loss_value).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

        if ema:
            ema.update(model_train)

        loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix(**{'loss': loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',
                    postfix=dict, mininterval=0.3)

    model_train_eval = ema.ema if ema else model_train
    model_train_eval.eval()
    detect_head = model_train_eval.model[-1] if hasattr(model_train_eval, 'model') else None
    detect_head_training = detect_head.training if detect_head is not None else None
    if detect_head is not None:
        detect_head.train()

    try:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            # ultralytics 数据集返回 uint8 (0-255)，模型需要 float32 [0,1]
            images = batch['img'].float() / 255.0
            # 拼接 ultralytics batch dict 为 Loss 期望的格式: [N, 6] = [batch_idx, cls, x, y, w, h]
            batch_targets = torch.cat([
                batch['batch_idx'].float().view(-1, 1),
                batch['cls'].float().view(-1, 1),
                batch['bboxes'].float(),
            ], dim=-1)
            with torch.no_grad():
                if cuda:
                    images = images.cuda(local_rank)
                    batch_targets = batch_targets.cuda(local_rank)
                optimizer.zero_grad()
                outputs = model_train_eval(images)
                loss_value = yolo_loss(outputs, batch_targets)

            val_loss += loss_value.item()
            if local_rank == 0:
                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
                pbar.update(1)
    finally:
        if detect_head is not None:
            detect_head.train(detect_head_training)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print(f'Epoch:{epoch + 1}/{Epoch}')
        print(f'Total Loss: {loss / epoch_step:.3f} || Val Loss: {val_loss / epoch_step_val:.3f}')

        # 保存 checkpoint
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict,
                       os.path.join(save_dir,
                                    f"ep{epoch + 1:03d}-loss{loss / epoch_step:.3f}"
                                    f"-val_loss{val_loss / epoch_step_val:.3f}.pth"))

        if len(loss_history.val_loss) <= 1 or \
           (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))
