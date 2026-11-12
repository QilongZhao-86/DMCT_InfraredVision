import os
import argparse
import yaml
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from omegaconf import OmegaConf
from dataset.image_floder import ImageFolder
from dataset.experiment_defaults import get_training_defaults
from infrared_model.base_model import make_adaptor
from infrared_model.NG import NG
# from infrared_model.de_net import DEResNet
from training.trainer_final import Trainer
from training import metricWrapper, seed_everything
from training.run_metadata import write_training_run_metadata
from infrared_model.Adapter import MultiScaleAdapter
import inspect
import tokenize
import io

def get_clean_source(func):
    source = inspect.getsource(func)
    result = []
    g = tokenize.generate_tokens(io.StringIO(source).readline)
    last_lineno = -1
    for token_type, token_string, (start_line, start_col), (end_line, end_col), line in g:
        if token_type == tokenize.COMMENT:
            continue
        if start_line > last_lineno:
            if line.strip() and not line.strip().startswith('#'):
                result.append(line.rstrip())
            last_lineno = start_line
    return "\n".join(result)

def init_ddp(args):
    if args.use_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)


def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def freeze_encoder_bn_eval(model):
    for m in model.modules():
        if isinstance(m, nn.SyncBatchNorm):
            m.eval()

def build_model(cfg):
    detector = make_adaptor(
        backbone_channel_list=cfg.backbone_channel_list,
        dense_low_channels=cfg.dense_low_channels,
        stages=cfg.stages,
        global_att_blocks=cfg.global_att_blocks,
        window_pos_embed_bkg_spatial_size=cfg.window_pos_embed_bkg_spatial_size,
        window_spec=cfg.window_spec,
        block=cfg.block,
        embed_dim=cfg.embed_dim,
        use_sam_decoder=cfg.use_sam_decoder,
        pe_inch=cfg.pe_inch,
        sam_ckpt_path=cfg.sam_ckpt_path,
    )
    return detector

def build_ng():
    ng_list = [
    'GaussianBlur', 
    'Posterize', 
    'add_new_striped_noise',
    'DownUpSample1',
    'LessContrast',
]
    net_ng = NG(ng_types=ng_list)
    return net_ng
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="NUDT-SIRST")
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--NGsave", type=bool, default=False)
    parser.add_argument("--NG_save_dir", default="./ng_save")
    parser.add_argument("--save_dir", default="./checkpoints/IRSTD-1k")
    parser.add_argument("--data_path", default="./dataset")
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--ng_config_path", default=None)
    parser.add_argument("--de_net_path", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--ng_start_epoch", type=int, default=25)
    parser.add_argument("--ng_end_epoch", type=int, default=None)
    parser.add_argument("--bilevel_start_epoch", type=int, default=25)
    parser.add_argument("--bilevel_end_epoch", type=int, default=None)
    parser.add_argument("--save_step", type=int, default=10)
    parser.add_argument("--use_ddp", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--eta_min",default=1e-6,type=float,help="Minimum learning rate for scheduler",)
    parser.add_argument("--loss_func",default="bceloss",type=str,help="Loss function to use (default: bceloss)",)
    parser.add_argument("--seed",default=2727,type=int,help="Random seed",)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers for train_final and its internal Trainer loaders")
    parser.add_argument("--note", default="Baseline", help="Experimental notes")
    parser.add_argument("--train_split_file", default=None)
    parser.add_argument("--val_split_file", default=None)
    parser.add_argument("--ng_strength_scale", type=float, default=0.5)

    args = parser.parse_args()
    dataset_defaults = get_training_defaults(args.dataset)
    args.image_size = args.image_size or dataset_defaults["image_size"]
    args.batch_size = args.batch_size or dataset_defaults["batch_size"]
    args.ng_end_epoch = args.ng_end_epoch or args.epoch
    args.bilevel_end_epoch = args.bilevel_end_epoch or args.epoch
    if args.num_workers < 0:
        parser.error("--num_workers must be non-negative")
    if not 1 <= args.ng_start_epoch <= args.ng_end_epoch <= args.epoch:
        parser.error("NG epoch range must be within the training epoch range")
    if not 1 <= args.bilevel_start_epoch <= args.bilevel_end_epoch <= args.epoch:
        parser.error("bilevel epoch range must be within the training epoch range")
    seed_everything(args.seed)
    if not args.train_split_file or not args.val_split_file:
        parser.error("Paper-protocol training requires explicit --train_split_file and --val_split_file")
    if args.train_split_file and args.ng_strength_scale < 0:
        parser.error("--ng_strength_scale must be non-negative")

    manifest = write_training_run_metadata(args, method="learned_ng_bilevel")
    print(f"Run metadata: {manifest}")

    init_ddp(args)

    device = torch.device("cuda", args.local_rank)

    if is_main_process():
        os.makedirs(args.save_dir, exist_ok=True)
        print("\n" + "="*40)
        print(f"{'EXPERIMENT CONFIG':^40}")
        print("="*40)
        args_dict = vars(args)
        for key in sorted(args_dict.keys()):
            print(f"{key:>25} : {args_dict[key]}")   
        print("="*40 + "\n")
        print("\n" + "="*20 + " 本次实验核心改动位置 " + "="*20)
        try:
            clean_code = get_clean_source(MultiScaleAdapter)
            print(clean_code)
        except Exception as e:
            print(f"无法获取源码: {e}")
        print("="*61 + "\n")

    train_set = ImageFolder(
        args.data_path,
        args.dataset,
        istraining=True,
        base_size=args.image_size,
        crop_size=args.image_size,
        split_file=args.train_split_file,
    )
    val_set = ImageFolder(
        args.data_path,
        args.dataset,
        istraining=False,
        base_size=args.image_size,
        crop_size=args.image_size,
        split_file=args.val_split_file,
    )

    train_sampler = DistributedSampler(train_set) if args.use_ddp else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if args.use_ddp else None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    ng_config=OmegaConf.load(args.ng_config_path)
    with open(args.config_path, "r") as f:
        adaptor_cfg = yaml.safe_load(f)
    net_model = make_adaptor(**adaptor_cfg,image_size=args.image_size)
    net_model._freeze_encoder()
    net_ng=build_ng()
    with torch.no_grad():
        net_ng.max_strength.mul_(args.ng_strength_scale)
    print('NG max_strength:', net_ng.max_strength.tolist())
    if args.use_ddp:
        net_model = nn.SyncBatchNorm.convert_sync_batchnorm(net_model)

    net_model._freeze_encoder()
    freeze_encoder_bn_eval(net_model)

    net_model.to(device)
    net_ng.to(device)

    if args.use_ddp:
        net_model = DDP(
            net_model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
        net_ng = DDP(
            net_ng,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
        )


    # net_de = None
    # if args.de_net_path is not None:
    #     print("you are loading DeResNet...")
    #     net_de = DEResNet(num_in_ch=3, num_degradation=2)
    #     net_de.load_model(args.de_net_path)
    #     print("Loadied DeResNet Successfully!")
    #     net_de.to(device)
    #     net_de.eval()
    #     for p in net_de.parameters():
    #         p.requires_grad_(False)
    loss_fn = nn.BCELoss(reduction="mean")
    print("loss fn 是BCELoss")
    print(next(net_model.parameters()).device)
    trainer = Trainer(
        model=net_model,
        train_dataset=train_set,
        val_dataset=val_set,
        ng=net_ng,
        loss_fn=loss_fn,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed=args.use_ddp,
        save_dir=args.save_dir,
        metric_wrapper=metricWrapper(),
        epoch=args.epoch,
        lr_model=args.lr,
    )
    trainable, total = 0, 0
    for n, p in trainer.model.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
                print("trainable:", n, p.shape)
    print(f"trainable params: {trainable}")
    print(f"total params: {total}")
    print(f"ratio: {trainable / total:.4%}")
    
    trainable, total = 0, 0
    for n, p in trainer.ng.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
                print("trainable in ng:", n, p.shape)
    print(f"trainable params: {trainable}")
    print(f"total params: {total}")
    print(f"ratio: {trainable / total:.4%}")
    from dataset.image_floder import NORMALIZATION
    if args.dataset == 'NUDT-SIRST':
        trainer.input_mean = [107.80905151367188/255]*3
        trainer.input_std = [33.02274703979492/255]*3
    else:
        trainer.input_mean, trainer.input_std = NORMALIZATION[args.dataset]
    trainer.ng_start_epoch = args.ng_start_epoch
    trainer.ng_end_epoch = args.ng_end_epoch
    start_epoch = 1
    best_val_miou = float("-inf")
    best_val_loss = float("inf")
    if args.resume is not None:
        resume_state = trainer.load_checkpoint(args.resume)
        start_epoch = int(trainer.epoch) + 1
        print(f"Resumed from epoch {trainer.epoch}; next epoch is {start_epoch}")

        # A periodic checkpoint carries the model/optimizer state but may predate
        # the selection metadata.  In that case read only the metadata from the
        # sibling best-validation checkpoint; do not replace the resumed model.
        if args.val_split_file:
            selection_state = resume_state
            if "selection_value" not in selection_state:
                best_path = os.path.join(args.save_dir, "best_internal_val.pth")
                if os.path.isfile(best_path) and os.path.abspath(best_path) != os.path.abspath(args.resume):
                    selection_state = torch.load(best_path, map_location="cpu")
            if "selection_value" in selection_state:
                best_val_miou = float(selection_state["selection_value"])
            elif selection_state.get("best_metric") is not None:
                best_val_miou = float(selection_state["best_metric"])
            if selection_state.get("validation_loss") is not None:
                best_val_loss = float(selection_state["validation_loss"])
    for epoch in range(start_epoch, args.epoch+1):
        trainer.epoch=epoch
        total_adv_epochs = args.epoch - args.ng_start_epoch
        current_step = epoch - args.ng_start_epoch
        progress = current_step / max(1, total_adv_epochs)  # 0~1

        alpha_min = 0.001
        alpha_peak = 0.2

        progress = current_step / max(1, total_adv_epochs)
        p_peak = 0.25

        progress = max(0.0, min(1.0, progress))

        drop_window = 0.02  # 从 0.25 到 0.30 迅速降到 alpha_min（可调）

        p_drop_end = min(1.0, p_peak + drop_window)

        if progress <= p_peak:
            # 0 -> 0.25: alpha_min -> alpha_peak（线性）
            t = progress / p_peak
            current_alpha_ng = alpha_min + (alpha_peak - alpha_min) * t

        elif progress <= p_drop_end:
            # 0.25 -> 0.25+drop_window: alpha_peak -> alpha_min（快速线性下降）
            t = (progress - p_peak) / max(1e-12, (p_drop_end - p_peak))  # 0~1
            current_alpha_ng = alpha_peak + (alpha_min - alpha_peak) * t

        else:
            # 后面保持不变
            current_alpha_ng = alpha_min
        current_alpha_ng = 1 -current_alpha_ng
        use_ng = args.ng_start_epoch <= epoch <= args.ng_end_epoch
        use_bilevel = args.bilevel_start_epoch <= epoch <= args.bilevel_end_epoch
        if use_ng and use_bilevel:
            avg_loss=trainer.train_one_epoch_adv_bilevel(alpha=current_alpha_ng,save=args.NGsave,ng_save_dir=args.NG_save_dir)
        else:
            avg_loss = trainer.train_one_epoch()
        print("train loss:",{avg_loss}) 
        if is_main_process():
            val_result = trainer.evaluate()
            if args.val_split_file and val_result is not None:
                val_loss, _ = val_result
                _, val_miou, _ = trainer.metric_wrapper.miou_meter.get()
                if (val_miou > best_val_miou) or (val_miou == best_val_miou and val_loss < best_val_loss):
                    best_val_miou, best_val_loss = val_miou, val_loss
                    trainer.best_metric = float(val_miou)
                    trainer.save_checkpoint(
                        os.path.join(args.save_dir, "best_internal_val.pth"),
                        extra={"selection_split": args.val_split_file,
                               "selection_metric": "mIoU",
                               "selection_value": float(val_miou),
                               "validation_loss": float(val_loss),
                               "epoch": int(epoch)},
                    )
        if is_main_process() and epoch % args.save_step == 0:
            trainer.save_checkpoint(
                os.path.join(args.save_dir, f"model_epoch_{epoch}.pth"),
                extra={"selection_split": args.val_split_file,
                       "selection_metric": "mIoU" if args.val_split_file else None,
                       "selection_value": best_val_miou,
                       "validation_loss": best_val_loss,
                       "epoch": int(epoch)},
            )        
if __name__ == "__main__":
    main()
