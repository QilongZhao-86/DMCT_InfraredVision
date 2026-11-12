import os
import time
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.image_floder import ImageFolder
from dataset.experiment_defaults import get_training_defaults
from metrics import PD_FAmeter, mIoUmeter, nIoUmeter
from infrared_model.base_model import make_adaptor
from training import seed_everything
from training.run_metadata import write_evaluation_metadata


def extract_state_dict(ckpt):
    """
    从不同格式的 checkpoint 中提取真正的模型参数
    """
    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint should be a dict, but got {type(ckpt)}")

    candidate_keys = [
        "model_state_dict",
        "state_dict",
        "model",
        "net",
        "ema_state_dict",
    ]

    for key in candidate_keys:
        if key in ckpt and isinstance(ckpt[key], dict):
            print(f"Use checkpoint key: '{key}'")
            return ckpt[key]

    # 如果最外层本身就是参数字典
    is_probably_state_dict = all(
        isinstance(k, str) for k in ckpt.keys()
    ) and any(isinstance(v, torch.Tensor) for v in ckpt.values())

    if is_probably_state_dict:
        print("Use raw checkpoint as state_dict")
        return ckpt

    raise KeyError(
        "Cannot find model weights in checkpoint. "
        f"Available keys: {list(ckpt.keys())}"
    )


def clean_state_dict(state_dict):
    """
    清理参数名前缀:
    - module.xxx        -> xxx
    - _orig_mod.xxx     -> xxx
    """
    new_state_dict = {}

    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]
        if new_k.startswith("_orig_mod."):
            new_k = new_k[len("_orig_mod."):]
        new_state_dict[new_k] = v

    return new_state_dict


def smart_load_state_dict(model, state_dict, strict=False):
    """
    只加载名字和 shape 都匹配的参数，避免因为少量不匹配直接报错
    """
    model_state = model.state_dict()
    filtered_state = {}
    skipped = []

    for k, v in state_dict.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                filtered_state[k] = v
            else:
                skipped.append(
                    f"{k}: ckpt{tuple(v.shape)} != model{tuple(model_state[k].shape)}"
                )

    missing_keys = [k for k in model_state.keys() if k not in filtered_state]
    unexpected_keys = [k for k in state_dict.keys() if k not in model_state]

    load_msg = model.load_state_dict(filtered_state, strict=strict)

    print(f"Loaded params: {len(filtered_state)}/{len(model_state)}")
    print(f"Missing params: {len(missing_keys)}")
    print(f"Unexpected params: {len(unexpected_keys)}")
    print(f"Shape mismatched params: {len(skipped)}")

    if len(unexpected_keys) > 0:
        print("First 20 unexpected keys:")
        for k in unexpected_keys[:20]:
            print(f"  {k}")

    if len(skipped) > 0:
        print("First 20 shape mismatched keys:")
        for s in skipped[:20]:
            print(f"  {s}")

    if hasattr(load_msg, "missing_keys") and len(load_msg.missing_keys) > 0:
        print("First 20 load_state_dict missing keys:")
        for k in load_msg.missing_keys[:20]:
            print(f"  {k}")

    if hasattr(load_msg, "unexpected_keys") and len(load_msg.unexpected_keys) > 0:
        print("First 20 load_state_dict unexpected keys:")
        for k in load_msg.unexpected_keys[:20]:
            print(f"  {k}")

    return load_msg


@torch.no_grad()
def evaluation(test_loader, predictor, device, save_pr_npz_path=None, output_type="logit", sample_per_image=1000, per_image_csv=None):
    predictor.eval()
    predictor = predictor.to(device)

    mIoU_meter = mIoUmeter()
    pdfa_meter = PD_FAmeter()
    nIoU_meter = nIoUmeter(1, 0.5)

    all_scores = []
    all_labels = []

    import csv
    records = []
    for _, (batch_data, gt_masks, names) in enumerate(test_loader):
        batch_data = batch_data.to(device, non_blocking=True)
        gt_masks = gt_masks.to(device, non_blocking=True)

        pred_out = predictor(batch_data.repeat(1, 3, 1, 1) if batch_data.dim() == 4 and batch_data.size(1) == 1 else batch_data)[0]

        if output_type == "logit":
            pred_logit = pred_out
            pred_prob = torch.sigmoid(pred_logit)
            pred_mask = pred_logit > 0
        else:
            pred_prob = pred_out
            pred_logit = pred_out
            pred_mask = pred_prob > 0.5

        B, _, H, W = batch_data.shape
        for i, name in enumerate(names):
            p = pred_mask[i].detach().cpu().bool()
            g = gt_masks[i].detach().cpu() > 0
            tp = (p & g).sum().item()
            fp = (p & ~g).sum().item()
            fn = (~p & g).sum().item()
            records.append([name, H, W, tp, fp, fn, tp/max(1,tp+fp+fn), 2*tp/max(1,2*tp+fp+fn)])

        nIoU_meter.update(
            pred_logit.reshape(B, 1, H, W).cpu().float(),
            gt_masks.reshape(B, 1, H, W).cpu().float(),
        )
        mIoU_meter.update(
            pred_mask.reshape(B, 1, H, W).cpu().float(),
            gt_masks.reshape(B, 1, H, W).cpu().float(),
        )
        pdfa_meter.update(
            pred_mask.reshape(B, H, W).cpu().float(),
            gt_masks.reshape(B, H, W).cpu().float(),
            [H, W],
        )

        if save_pr_npz_path is not None and sample_per_image > 0:
            prob_cpu = pred_prob.detach().cpu().reshape(B, -1).numpy().astype(np.float32)
            gt_cpu = (gt_masks.detach().cpu().reshape(B, -1).numpy() > 0.5).astype(np.uint8)

            for b in range(B):
                n = prob_cpu[b].shape[0]
                k = min(sample_per_image, n)
                idx = np.random.choice(n, size=k, replace=False)
                all_scores.append(prob_cpu[b][idx])
                all_labels.append(gt_cpu[b][idx])

    if save_pr_npz_path is not None:
        os.makedirs(os.path.dirname(save_pr_npz_path) or ".", exist_ok=True)
        scores = np.concatenate(all_scores, axis=0) if all_scores else np.array([], dtype=np.float32)
        labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([], dtype=np.uint8)
        np.savez_compressed(save_pr_npz_path, scores=scores, labels=labels)
        print(f"Saved PR data to: {save_pr_npz_path}, samples={len(scores)}")

    if per_image_csv:
        os.makedirs(os.path.dirname(os.path.abspath(per_image_csv)), exist_ok=True)
        with open(per_image_csv, 'w', newline='') as handle:
            writer=csv.writer(handle)
            writer.writerow(['image','height','width','tp','fp','fn','IoU','F1'])
            writer.writerows(records)
    return mIoU_meter, pdfa_meter, nIoU_meter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dataset", default="IRSTD-1K", type=str)
    parser.add_argument("--image_size", default=None, type=int)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--data_path", default="./dataset", type=str)
    parser.add_argument("--weights", default="./checkpoints/IRSTD-1k.pt", type=str)
    parser.add_argument("--device", default="cuda:1", type=str)
    parser.add_argument("--seed", default=2727, type=int)
    parser.add_argument("--num_workers", default=7, type=int)

    parser.add_argument("--save_pr_npz", default="", type=str)
    parser.add_argument("--output_type", default="logit", choices=["logit", "prob"])
    parser.add_argument("--sample_per_image", default=1000, type=int)
    parser.add_argument("--test_split_file", required=True, type=str)

    parser.add_argument("--per_image_csv", default="")
    args = parser.parse_args()
    dataset_defaults = get_training_defaults(args.dataset)
    args.image_size = args.image_size or dataset_defaults["image_size"]
    args.batch_size = args.batch_size or dataset_defaults["batch_size"]

    output_dir = (
        os.path.dirname(os.path.abspath(args.per_image_csv))
        if args.per_image_csv
        else os.path.join(os.getcwd(), "outputs", "evaluation")
    )
    manifest = write_evaluation_metadata(args, output_dir=Path(output_dir))
    print(f"Evaluation metadata: {manifest}")

    seed_everything(args.seed)

    t0 = time.time()
    predictor = make_adaptor(
        backbone_channel_list=[384, 192, 96],
        stages=[1, 2, 7],
        global_att_blocks=[5, 7, 9],
        block="res",
        embed_dim=96,
        dense_low_channels=[96, 48, 24],
        window_spec=(8, 4, 14),
        use_sam_decoder=True,
    )
    print(f"build_model cost: {time.time() - t0:.2f}s")

    t1 = time.time()
    test_set = ImageFolder(
        args.data_path, args.dataset,
        istraining=False,
        base_size=args.image_size,
        crop_size=args.image_size,
        split_file=args.test_split_file,
    )
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers
    )
    print(f"make_loader cost: {time.time() - t1:.2f}s")

    t2 = time.time()
    ckpt = torch.load(args.weights, map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ["model_state_dict", "model_state", "state_dict", "model"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                state_dict = ckpt[k]
                print(f"Using ckpt['{k}'] as state_dict")
                break
        else:
            if any(torch.is_tensor(v) for v in ckpt.values()):
                state_dict = ckpt
                print("Using ckpt as direct state_dict")
            else:
                raise ValueError(f"Checkpoint keys look not like state_dict: {list(ckpt.keys())[:20]}")
    else:
        raise ValueError("Checkpoint is not a dict, unexpected format.")

    missing, unexpected = predictor.load_state_dict(state_dict, strict=True)
    print("[Load Checkpoint]")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))
    print("Sample missing:", missing[:20])
    print("Sample unexpected:", unexpected[:20])


    t3 = time.time()
    mIoU_meter, pdfa_meter, nIoU_meter = evaluation(
        test_loader,
        predictor,
        args.device,
        save_pr_npz_path=args.save_pr_npz if args.save_pr_npz else None,
        output_type=args.output_type,
        sample_per_image=args.sample_per_image,
        per_image_csv=args.per_image_csv,
    )
    print(f"evaluation cost: {time.time() - t3:.2f}s")

    _, mIoU, fscore = mIoU_meter.get()
    nIoU = nIoU_meter.get()[1]
    pd, fa = pdfa_meter.get()

    print(
        f"mIoU = {mIoU*1e2:.2f} | "
        f"F1Score = {fscore*1e2:.2f} | "
        f"nIoU = {nIoU*1e2:.2f} | "
        f"Pd = {pd*1e2:.2f} | "
        f"Fa = {fa*1e6:.2f}"
    )


if __name__ == "__main__":
    main()
