import os
from typing import Optional, Dict, Any, Tuple, Union
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from loguru import logger
import enlighten
import torch.nn.functional as F
from .metrics_config import metricWrapper
# from .blo_strategy import one_step_bilevel_fd
# from tqdm import tqdm
from .Adan import Adan
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        *,
        ng: Optional[nn.Module] = None,
        loss_fn: Optional[nn.Module] = None,
        device: Union[str, torch.device] = "cuda",
        batch_size: int = 8,
        num_workers: int = 4,
        distributed: bool = False,
        save_dir: str = "./checkpoints",
        metric_wrapper: Optional[metricWrapper] = metricWrapper(),
        lr_model: float = 0.01 ,
        epoch: int = 300,
        eta_min: float=1e-6,
    ):
        self.distributed=distributed
        self.device = device
        if distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        self.model=model##模型含有adapter
        self.model = self.model.to(self.device)
        self.ng=ng#复合降质网络
        self.ng = ng.to(self.device) if ng is not None else None
        self.loss_fn = loss_fn if loss_fn is not None else nn.BCELoss()
        self.save_dir = save_dir
        self.metric_wrapper = metric_wrapper
        os.makedirs(self.save_dir, exist_ok=True)
        self.epoch = epoch
        self.best_metric = 0.0
        
        
        self.upper_params = list(self.model.image_encoder.hl.parameters())
        upper_ids = {id(p) for p in self.upper_params}
        self.ng_params = list(self.ng.parameters()) if self.ng is not None else []
        self.lower_params = [
            p for p in self.model.parameters() 
            if p.requires_grad and id(p) not in upper_ids
        ]

        self.optimizer_ng = torch.optim.SGD(self.ng_params, lr=lr_model*0.01, weight_decay=1e-6)
        self.optimizer_lower = Adan(self.lower_params, lr=lr_model, weight_decay=1e-4)
        self.optimizer_upper = Adan(self.upper_params, lr=lr_model * 0.5, weight_decay=1e-4)

        self.sched_ng = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_ng, T_max=epoch, eta_min=eta_min
        )
        self.sched_lower = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_lower, T_max=epoch, eta_min=eta_min
        )
        self.sched_upper = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_upper, T_max=epoch, eta_min=eta_min
        )
        self.train_sampler = DistributedSampler(train_dataset) if distributed else None
        self.val_sampler = (
            DistributedSampler(val_dataset)
            if (distributed and val_dataset is not None)
            else None
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(self.train_sampler is None),
            num_workers=num_workers,
            pin_memory=True,
            sampler=self.train_sampler,
        )
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                sampler=self.val_sampler,
            )

    def train_one_epoch(self) -> float:
        """
        最基础的端到端训练：只更新 model (分为 upper 和 lower)，不涉及 ng 降质网络，也不使用双层优化策略。
        """
        self.model.train()
        total_loss_val = 0
        total_samples = 0
        
        if self.rank == 0:
            manager = enlighten.get_manager()
            pbar = manager.counter(
                total=len(self.train_loader), 
                desc=f"[Epoch {self.epoch:3d}] Model-Only Training", 
                unit="batch",
                color="cyan"
            )

        for idx, batch in enumerate(self.train_loader):
            img, mask, _ = [b.to(self.device) if torch.is_tensor(b) else b for b in batch]

            self.optimizer_lower.zero_grad(set_to_none=True)
            self.optimizer_upper.zero_grad(set_to_none=True)

            pred_logits = self.model(img)
            
            loss = 0
            if isinstance(pred_logits, (list, tuple)):
                for pred_logit in pred_logits:
                    loss += self.loss_fn(pred_logit.sigmoid(), mask)
            else:
                loss = self.loss_fn(pred_logits.sigmoid(), mask)

            loss.backward()
            
            self.optimizer_lower.step()
            self.optimizer_upper.step()

            total_loss_val += loss.item() * img.size(0)
            total_samples += img.size(0)
            
            if self.rank == 0:
                pbar.update()
                pbar.desc = f"[End to End training] Epoch {self.epoch} Loss: {loss.item():.6f}"

        if self.rank == 0:
            pbar.close()

        self.sched_lower.step()
        self.sched_upper.step()

        return total_loss_val / total_samples 
    
    
    
    def clean_loss(self,preds, masks):
        """计算loss，clean的"""
        return sum(self.loss_fn(p.sigmoid(), masks) for p in preds)
    
    def ng_loss(self, img, mask,save,ng_save_dir):
        """计算降质图像下的loss并且实时保存当前降质图像"""
        img_ng = self.ng(img)
        pred_ng = self.model(img_ng)
        bs = img.size(0)
        # if save and self.rank == 0 :
        #         with torch.no_grad():
        #             for i in range(min(bs, 4)): 
        #                 save_name_orig = os.path.join(ng_save_dir, f"{i:06d}_orig.png")
        #                 save_name_deg = os.path.join(ng_save_dir, f"{i:06d}_deg.png")
        #                 save_image(img[i], save_name_orig, normalize=True)
        #                 save_image(img_ng[i], save_name_deg, normalize=True)
        
        if isinstance(pred_ng, (list, tuple)):
            return sum(self.loss_fn(p.sigmoid(), mask) for p in pred_ng)
        return self.loss_fn(pred_ng.sigmoid(), mask)
        
    def train_one_epoch_ng(self, alpha: float = 0.5, save: bool = False, ng_save_dir: str = "./ng_save") -> float:
        """进行降质对抗训练"""
        self.model.train()
        self.ng.train()
        total_loss_val = 0
        total_samples = 0
        pbar = None
        if self.rank == 0:
            manager = enlighten.get_manager()
            pbar = manager.counter(
                total=len(self.train_loader), 
                desc=f"[Epoch {self.epoch:3d}] Adv-Training", 
                unit="batch",
                color="red"
            )

        for idx, batch in enumerate(self.train_loader):
            img, mask, _ = [b.to(self.device) if torch.is_tensor(b) else b for b in batch]

            self.optimizer_ng.zero_grad(set_to_none=True)
            
            loss_ng = self.ng_loss(img, mask,save,ng_save_dir)##对抗的loss
            (-loss_ng).backward() 
            self.optimizer_ng.step()

            self.optimizer_lower.zero_grad(set_to_none=True)
            self.optimizer_upper.zero_grad(set_to_none=True)
            
            img_ng_new = self.ng(img).detach() 
            pred_ng = self.model(img_ng_new)
            pred_clean = self.model(img)
            
            loss_clean = self.clean_loss(pred_clean, mask)
            loss_ng = self.clean_loss(pred_ng, mask) 
            
            loss_model = alpha * loss_clean + (1 - alpha) * loss_ng
            loss_model.backward()
            
            self.optimizer_lower.step()
            self.optimizer_upper.step()
            
            current_loss = loss_model.item()
            total_loss_val += loss_model.item() * img.size(0)
            total_samples += img.size(0)
            if self.rank == 0:
                if pbar is not None:
                    pbar.update()
                    pbar.desc = f"[Adv-Ep {self.epoch}] L_ng:{loss_ng.item():.3f} | L_mod:{current_loss:.3f}"
                
            if self.rank == 0 and idx % 20 == 0:
                print(f"[Epoch {self.epoch}][Batch {idx}] L_adv_for_ng: {loss_ng.item():.4f} | L_model: {loss_model.item():.4f}")
        if self.rank == 0 and pbar is not None:
            pbar.close()
        self.sched_ng.step()
        self.sched_lower.step()
        self.sched_upper.step()

        return total_loss_val / total_samples
    
    

    def train_one_epoch_blo(self, beta=0.1, delta=1e-8, tau=2e-4):
        """只进行双层优化训练的：y_params: lower_params (ω) - 下层、 x_params: upper_params (θ) - 上层
        """
        self.model.train()
        stats = {
            "norm_f": [], "norm_h": [], "lambda_k": [],
            "loss_diff": []
        }

        train_iter = iter(self.train_loader)
        img_tr_old, mask_tr_old = None, None

        for step, batch_tr in enumerate(train_iter):
            img_tr, mask_tr, _ = [b.to(self.device) if torch.is_tensor(b) else b for b in batch_tr]
            
            if step == 0:
                try:
                    batch_va = next(train_iter)
                    img_va, mask_va, _ = [b.to(self.device) if torch.is_tensor(b) else b for b in batch_va]
                except StopIteration: break
            else:
                img_va, mask_va = img_tr_old, mask_tr_old

            img_tr_old, mask_tr_old = img_tr, mask_tr
            if img_va is None: continue

            x_params = self.upper_params
            y_params = [p for p in self.lower_params if p.requires_grad]
            combined_params = list(x_params) + list(y_params)

            self.optimizer_upper.zero_grad(set_to_none=True)
            self.optimizer_lower.zero_grad(set_to_none=True)

            pred_tr = self.model(img_tr)
            g = self.clean_loss(pred_tr, mask_tr)
            v_raw = torch.autograd.grad(g, y_params, allow_unused=True)

            y_active, v_active = [], []
            for p, vi in zip(y_params, v_raw):
                if vi is not None:
                    y_active.append(p)
                    v_active.append(vi.detach())

            if not y_active: continue
            
            pred_va = self.model(img_va)
            f = self.clean_loss(pred_va, mask_va)
            grads_f_all = torch.autograd.grad(f, combined_params, allow_unused=True)


            tau = 2e-4
            # w + tau * v
            with torch.no_grad():
                for p, v in zip(y_active, v_active): p.add_(tau * v)
            g_p = self.clean_loss(self.model(img_tr), mask_tr)
            grads_p = torch.autograd.grad(g_p, combined_params, allow_unused=True)

            # w - tau * v
            with torch.no_grad():
                for p, v in zip(y_active, v_active): p.add_(-2.0 * tau * v)
            g_m = self.clean_loss(self.model(img_tr), mask_tr)
            grads_m = torch.autograd.grad(g_m, combined_params, allow_unused=True)

            with torch.no_grad():
                for p, v in zip(y_active, v_active): p.add_(tau * v)

            f_list, h_list = [], []
            for i, p in enumerate(combined_params):
                gf = grads_f_all[i].detach() if grads_f_all[i] is not None else torch.zeros_like(p)
                gp = grads_p[i].detach() if grads_p[i] is not None else torch.zeros_like(p)
                gm = grads_m[i].detach() if grads_m[i] is not None else torch.zeros_like(p)

                gh = (gp - gm) / (2.0 * tau)
                f_list.append(gf)
                h_list.append(gh)

            flat_f = torch.cat([g.reshape(-1) for g in f_list])
            flat_h = torch.cat([g.reshape(-1) for g in h_list])
            n_f = torch.norm(flat_f)
            n_h = torch.norm(flat_h)

            lambda_val = torch.clamp(beta * (n_f / (n_h + delta)), min=0.001, max=1.0)

            for i, p in enumerate(combined_params):
                p.grad = f_list[i] + lambda_val * h_list[i]

            self.optimizer_upper.step()
            self.optimizer_lower.step()

            loss_diff = torch.abs(g_p - g_m).item()
            stats["norm_f"].append(n_f.item())
            stats["norm_h"].append(n_h.item())
            stats["lambda_k"].append(lambda_val.item())
            stats["loss_diff"].append(loss_diff)

            if self.rank == 0 and step % 20 == 0:
                print(f"[BLO Step {step:4d}] λ: {lambda_val:.4f} | f_norm: {n_f:.2e} | L_diff: {loss_diff:.2e}")

        self.sched_lower.step()
        self.sched_upper.step()

        if self.rank == 0:
            avg_lambda = sum(stats['lambda_k']) / len(stats['lambda_k'])
            print(f"\n>>> [Epoch {self.epoch}] BLO Avg λ: {avg_lambda:.4f}")

        return sum(stats["norm_f"]) / len(stats["norm_f"])
    
    def train_one_epoch_adv_bilevel(self, alpha=0.5, beta=0.1, delta=1e-8,save=False,ng_save_dir="./ng_save"):
        """
        对抗降质&双层优化
        """
        self.model.train()
        self.ng.train()

        stats = {"lambda_k": [], "norm_f": [], "l_ng": [], "l_mod": []}
        train_iter = iter(self.train_loader)
        img_tr_old, mask_tr_old = None, None
        num_steps = len(self.train_loader)

        pbar = None
        if self.rank == 0:
            pbar = enlighten.get_manager().counter(total=num_steps, desc=f"ADV-BLO Ep {self.epoch}", color="gold")

        for step, batch_tr in enumerate(train_iter):
            img_tr, mask_tr, _ = [b.to(self.device) if torch.is_tensor(b) else b for b in batch_tr]

            self.optimizer_ng.zero_grad(set_to_none=True)
            l_phi = self.ng_loss(img_tr, mask_tr,save,ng_save_dir) 
            (-l_phi).backward()
            self.optimizer_ng.step()

            if img_tr_old is None:
                img_tr_old, mask_tr_old = img_tr, mask_tr
                if self.rank == 0 and pbar:
                    pbar.update()
                    pbar.desc = f"[Ep{self.epoch}] (warmup) L_phi:{l_phi.item():.3f}"
                continue

            img_va, mask_va = img_tr_old, mask_tr_old
            img_tr_old, mask_tr_old = img_tr, mask_tr

            x_params = self.upper_params
            y_params = [p for p in self.lower_params if p.requires_grad]
            combined_params = list(x_params) + list(y_params)

            self.optimizer_upper.zero_grad(set_to_none=True)
            self.optimizer_lower.zero_grad(set_to_none=True)
            self.ng.eval()
            # with torch.no_grad():
            #     img_ng_tr = self.ng(img_tr)

            l_tr = self.clean_loss(self.model(img_tr), mask_tr)

            v_raw = torch.autograd.grad(l_tr, y_params, allow_unused=True)
            y_active = [p for p, v in zip(y_params, v_raw) if v is not None]
            v_active = [v.detach() for v in v_raw if v is not None]

            with torch.no_grad():
                img_va_ng = self.ng(img_va)

            l_va = alpha *self.clean_loss(self.model(img_va), mask_va)+(1 - alpha) * self.clean_loss(self.model(img_va_ng), mask_va)

            grads_f_all = torch.autograd.grad(l_va, combined_params, allow_unused=True)

            tau = 2e-4
            # w + tau * v
            with torch.no_grad():
                for p, v in zip(y_active, v_active): p.add_(tau * v)
            l_p =  self.clean_loss(self.model(img_tr), mask_tr)
            grads_p = torch.autograd.grad(l_p, combined_params, allow_unused=True)

            # w - tau * v
            with torch.no_grad():
                for p, v in zip(y_active, v_active): p.add_(-2.0 * tau * v)
            l_m = self.clean_loss(self.model(img_tr), mask_tr)
            grads_m = torch.autograd.grad(l_m, combined_params, allow_unused=True)

            with torch.no_grad():
                for p, v in zip(y_active, v_active): p.add_(tau * v)

            f_list, h_list = [], []
            for i, p in enumerate(combined_params):
                gf = grads_f_all[i].detach() if grads_f_all[i] is not None else torch.zeros_like(p)
                gp = grads_p[i].detach() if grads_p[i] is not None else torch.zeros_like(p)
                gm = grads_m[i].detach() if grads_m[i] is not None else torch.zeros_like(p)
                
                gh = (gp - gm) / (2.0 * tau)
                f_list.append(gf)
                h_list.append(gh)

            flat_f = torch.cat([g.reshape(-1) for g in f_list])
            flat_h = torch.cat([g.reshape(-1) for g in h_list])
            n_f, n_h = torch.norm(flat_f), torch.norm(flat_h)
            
            lambda_val = torch.clamp(beta * (n_f / (n_h + delta)), min=0.001, max=0.1)

            for i, p in enumerate(combined_params):
                p.grad = f_list[i] + lambda_val * h_list[i]

            self.optimizer_upper.step()
            self.optimizer_lower.step()

            if self.rank == 0:
                if pbar:
                    pbar.update()
                    pbar.desc = f"[Ep{self.epoch}] λ:{lambda_val:.3f} | L_phi:{l_phi.item():.3f}"
                if step % 20 == 0:
                    print(f"[Step {step}] λ:{lambda_val:.4f} | f_norm:{n_f:.2e} | L_ng:{l_phi.item():.4f}")

        self.sched_ng.step()
        self.sched_lower.step()
        self.sched_upper.step()

        if self.rank == 0 and pbar: pbar.close()
        return n_f.item() 
        
    def train_one_epoch_adv_lower(self, save=False, ng_save_dir="./ng_save"):
        self.model.train()
        self.ng.train()

        num_steps = len(self.train_loader)
        pbar = None
        if self.rank == 0:
            pbar = enlighten.get_manager().counter(
                total=num_steps,
                desc=f"LOWER Ep {self.epoch}",
                color="green"
            )

        last_l_phi = 0.0
        last_l_tr = 0.0

        for step, batch_tr in enumerate(self.train_loader):
            img_tr, mask_tr, _ = [
                b.to(self.device) if torch.is_tensor(b) else b
                for b in batch_tr
            ]

            # ---------------------------
            # 1) 更新 NG
            # ---------------------------
            self.optimizer_ng.zero_grad(set_to_none=True)
            l_phi = self.ng_loss(img_tr, mask_tr, save, ng_save_dir)
            (-l_phi).backward()
            self.optimizer_ng.step()

            # ---------------------------
            # 2) 只更新 lower_params
            # ---------------------------
            self.optimizer_lower.zero_grad(set_to_none=True)

            pred_tr = self.model(img_tr)
            l_tr = self.clean_loss(pred_tr, mask_tr)
            l_tr.backward()

            self.optimizer_lower.step()

            last_l_phi = l_phi.item()
            last_l_tr = l_tr.item()

            if self.rank == 0 and pbar:
                pbar.update()
                pbar.desc = f"[Ep{self.epoch}] LOWER | L_phi:{last_l_phi:.3f} | L_tr:{last_l_tr:.3f}"

            if self.rank == 0 and step % 20 == 0:
                print(f"[LOWER][Step {step}] L_phi:{last_l_phi:.4f} | L_tr:{last_l_tr:.4f}")

        self.sched_ng.step()
        self.sched_lower.step()

        if self.rank == 0 and pbar:
            pbar.close()

        return last_l_tr
    
    
    def train_one_epoch_adv_upper(self):
        self.model.train()
        if self.ng is not None:
            self.ng.eval()

        num_steps = len(self.train_loader)
        pbar = None
        if self.rank == 0:
            pbar = enlighten.get_manager().counter(
                total=num_steps,
                desc=f"UPPER-LITE Ep {self.epoch}",
                color="gold"
            )

        img_tr_old, mask_tr_old = None, None
        last_l_va = 0.0

        x_params = [p for p in self.upper_params if p.requires_grad]
        y_params = [p for p in self.lower_params if p.requires_grad]

        # 先冻结 lower，减少显存
        lower_requires_grad = [p.requires_grad for p in y_params]
        for p in y_params:
            p.requires_grad_(False)

        for step, batch_tr in enumerate(self.train_loader):
            img_tr, mask_tr, _ = [
                b.to(self.device) if torch.is_tensor(b) else b
                for b in batch_tr
            ]

            if img_tr_old is None:
                img_tr_old, mask_tr_old = img_tr, mask_tr
                if self.rank == 0 and pbar:
                    pbar.update()
                    pbar.desc = f"[Ep{self.epoch}] UPPER-LITE | warmup"
                continue

            img_va, mask_va = img_tr_old, mask_tr_old
            img_tr_old, mask_tr_old = img_tr, mask_tr

            self.optimizer_upper.zero_grad(set_to_none=True)

            pred_va = self.model(img_va)
            l_va = self.clean_loss(pred_va, mask_va)

            l_va.backward()
            self.optimizer_upper.step()

            last_l_va = l_va.item()

            if self.rank == 0 and pbar:
                pbar.update()
                pbar.desc = f"[Ep{self.epoch}] UPPER-LITE | L_va:{last_l_va:.3f}"

            if self.rank == 0 and step % 20 == 0:
                print(f"[UPPER-LITE][Step {step}] L_va:{last_l_va:.4f}")

        # 恢复 lower 的 requires_grad
        for p, rg in zip(y_params, lower_requires_grad):
            p.requires_grad_(rg)

        self.sched_upper.step()

        if self.rank == 0 and pbar:
            pbar.close()

        return last_l_va
    
    @torch.no_grad()
    def get_degradation_stats(self, img_input):
        """
        统计一个 batch 上每种退化的平均信息：
        - avg_weight: feature_extractor 分配给该退化的平均权重
        - avg_gamma: 该退化的平均 gamma
        - effective_strength: avg_weight * avg_gamma
        """
        device = img_input.device
        batch_size = img_input.size(0)

        weights_type = self.feature_extractor(img_input)
        weights_type = weights_type.view(batch_size, 5, self.type_num)
        weights_type = torch.softmax(weights_type, dim=1)

        # 每张图，对 5 个 stage 求平均，得到该图对每种退化类型的权重
        avg_w = weights_type.mean(dim=1)   # [B, type_num]

        # 按照 forward 里的逻辑重新采样 gamma
        gamma_tensor = torch.zeros(batch_size, self.type_num, device=device, dtype=avg_w.dtype)
        for i in range(batch_size):
            seed = random.random()
            gammas = self.split_gamma(seed, len(self.ng_types))
            gamma_tensor[i] = torch.tensor(gammas, device=device, dtype=avg_w.dtype)

        mean_w = avg_w.mean(dim=0)            # [type_num]
        mean_gamma = gamma_tensor.mean(dim=0) # [type_num]
        eff_strength = mean_w * mean_gamma

        stats = {}
        for ind, ng_type in enumerate(self.ng_types):
            item = {
                "avg_weight": mean_w[ind].item(),
                "avg_gamma": mean_gamma[ind].item(),
                "effective_strength": eff_strength[ind].item(),
            }

            # 额外补充更直观的物理强度参数
            if ng_type == "GaussianBlur":
                item["sigma"] = mean_gamma[ind].item() * 0.1

            elif ng_type == "Posterize":
                gamma_val = mean_gamma[ind].item()
                bits = int(round(8 - gamma_val))
                bits = max(4, min(8, bits))
                item["bits"] = bits

            elif ng_type == "add_new_striped_noise":
                gamma_val = mean_gamma[ind].item()
                intensity = min(0.01 * gamma_val, 0.05)
                item["stripe_intensity"] = intensity

            elif ng_type == "LessContrast":
                gamma_val = mean_gamma[ind].item()
                contrast_factor = 1.0 - gamma_val * (1.0 - 0.5)
                item["contrast_factor"] = contrast_factor

            elif ng_type == "DownUpSample1":
                gamma_val = mean_gamma[ind].item()
                scale = max(0.6, 1.0 - gamma_val * 0.1)
                item["scale"] = scale

            stats[ng_type] = item

        return stats
    def _get_ng_module(self):
        if self.ng is None:
            return None
        return self.ng.module if hasattr(self.ng, "module") else self.ng

    def init_ng_epoch_stats(self):
        ng_module = self._get_ng_module()
        if ng_module is None:
            self.ng_epoch_stats = {}
            return

        self.ng_epoch_stats = {
            name: {
                "weight_sum": 0.0,
                "gamma_sum": 0.0,
                "eff_sum": 0.0,
                "count": 0,
            }
            for name in ng_module.ng_types
        }

    @torch.no_grad()
    def update_ng_epoch_stats(self, img_batch):
        ng_module = self._get_ng_module()
        if ng_module is None:
            return
        if not hasattr(ng_module, "get_degradation_stats"):
            return

        stats = ng_module.get_degradation_stats(img_batch)

        for ng_name, info in stats.items():
            self.ng_epoch_stats[ng_name]["weight_sum"] += info["avg_weight"]
            self.ng_epoch_stats[ng_name]["gamma_sum"] += info["avg_gamma"]
            self.ng_epoch_stats[ng_name]["eff_sum"] += info["effective_strength"]
            self.ng_epoch_stats[ng_name]["count"] += 1

    def print_ng_epoch_stats(self):
        ng_module = self._get_ng_module()
        if ng_module is None:
            return
        if not hasattr(self, "ng_epoch_stats"):
            return

        print("\n" + "=" * 100)
        print(f"[Epoch {self.epoch}] NG Degradation Statistics (Epoch Average)")
        print("-" * 100)

        for ng_name in ng_module.ng_types:
            buf = self.ng_epoch_stats.get(ng_name, None)
            if buf is None:
                continue

            c = max(buf["count"], 1)
            avg_w = buf["weight_sum"] / c
            avg_g = buf["gamma_sum"] / c
            avg_eff = buf["eff_sum"] / c

            msg = (
                f"{ng_name:>24s} | "
                f"weight={avg_w:.4f} | "
                f"gamma={avg_g:.4f} | "
                f"eff={avg_eff:.4f}"
            )

            if ng_name == "GaussianBlur":
                msg += f" | sigma≈{avg_g * 0.1:.4f}"

            elif ng_name == "Posterize":
                bits = int(round(8 - avg_g))
                bits = max(4, min(8, bits))
                msg += f" | bits≈{bits}"

            elif ng_name == "add_new_striped_noise":
                intensity = min(0.01 * avg_g, 0.05)
                msg += f" | stripe≈{intensity:.6f}"

            elif "LessContrast" in ng_name:
                contrast_factor = 1.0 - avg_g * (1.0 - 0.5)
                msg += f" | contrast≈{contrast_factor:.4f}"

            elif "DownUpSample" in ng_name:
                scale = max(0.6, 1.0 - avg_g * 0.1)
                msg += f" | scale≈{scale:.4f}"

            print(msg)

        print("=" * 100 + "\n")
        
    @torch.no_grad()
    def evaluate(self) -> Optional[Tuple[float, Optional[metricWrapper]]]:
        """
        Evaluate the model on the validation dataset.
            
        Returns:
            tuple: (average_loss, metric_wrapper) or None if no validation data
        """
        if self.metric_wrapper is not None:
            self.metric_wrapper.reset()
        if self.rank == 0:
            manager = enlighten.get_manager()
            pbar = manager.counter(
                total=len(self.val_loader),
                desc=f"[Epoch {self.epoch: 3d}] Evaluating",
                unit="batch",
                color="red",
                leave=False,
            )
        self.model.eval()
        if self.val_loader is None:
            return None
        total_loss = 0
        total_samples = 0
        for idx, batch in enumerate(self.val_loader):
            batch_data, batch_masks, _ = batch
            batch_data = batch_data.to(self.device)
            batch_masks = batch_masks.to(self.device)
            pred_logit = self.model(batch_data)[0]
            loss = 0
            loss += self.loss_fn(pred_logit.sigmoid(), batch_masks)
            total_loss += loss.item() * batch_data.size(0)
            total_samples += batch_data.size(0)
            if self.rank == 0:
                pbar.update()
                pbar.desc = f"[Evaluating] Epoch {self.epoch:3d} loss={total_loss / total_samples: .6f}"
            if self.metric_wrapper:
                self.metric_wrapper(pred_logit, batch_masks)
        avg_loss = total_loss / total_samples
        if self.rank == 0:
            logger.info(
                f"[Eval][Epoch {self.epoch}] avg_loss={avg_loss:.6f} metrics: {self.metric_wrapper}"
            )
        return avg_loss, self.metric_wrapper
    def save_checkpoint(self, save_path: str, extra: Optional[Dict[str, Any]] = None) -> None:
        
        if self.rank != 0:
            return
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        def get_state(module):
            if module is None:
                return None
            if hasattr(module, "module"):
                return module.module.state_dict()
            return module.state_dict()


        checkpoint = {
            "epoch": getattr(self, "epoch", 0),
            "best_metric": getattr(self, "best_metric", 0.0),
            "model_state": get_state(getattr(self, "model", None)),
            "ng_state": get_state(getattr(self, "ng", None)),
            "hl_state": get_state(getattr(self, "upper_module", None)),
            "adapter_state": get_state(getattr(self, "lower_module", None)),

            "optimizer_state": self.optimizer.state_dict() if hasattr(self, "optimizer") and self.optimizer else None,
            "ng_optimizer_state": self.ng_optimizer.state_dict() if hasattr(self, "ng_optimizer") and self.ng_optimizer else None,
            "upper_optimizer_state": self.upper_optimizer.state_dict() if hasattr(self, "upper_optimizer") and self.upper_optimizer else None,
            "lower_optimizer_state": self.lower_optimizer.state_dict() if hasattr(self, "lower_optimizer") and self.lower_optimizer else None,
            "scheduler_state": self.scheduler.state_dict() if hasattr(self, "scheduler") and self.scheduler else None,
        }

        if extra is not None:
            checkpoint.update(extra)

        try:
            torch.save(checkpoint, save_path)
            if self.rank == 0:
                print(f"Checkpoint saved successfully at: {save_path}")
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")

    # def save_checkpoint(self, save_path: str, extra: Optional[Dict[str, Any]] = None) -> None:
    #     """
    #     Save training checkpoint to disk.
        
    #     Only rank 0 process saves checkpoints in distributed training.
        
    #     Args:
    #         save_path: Path to save the checkpoint
    #         extra: Additional data to include in checkpoint (optional)
    #     """
    #     if self.rank != 0:
    #         return
    #     if not os.path.exists(self.save_dir):
    #         os.makedirs(self.save_dir)
    #     state = {
    #         "epoch": self.epoch,
    #         "model_state_dict": self.model.module.state_dict()
    #         if self.distributed
    #         else self.model.state_dict(),
    #         "optimizer_state_dict": self.optimizer.state_dict(),
    #         "scheduler_state_dict": self.scheduler.state_dict()
    #         if self.scheduler
    #         else None,
    #         "best_metric": self.best_metric,
    #     }
    #     if extra:
    #         state.update(extra)
    #     torch.save(state, save_path)
        
    def save_ng_checkpoint(
        self,
        save_path: str,
        ng_model: nn.Module,
        ng_optimizer: torch.optim.Optimizer,
        ng_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save NG (Noise Generator / NGModel) checkpoint.

        Only rank 0 process saves checkpoints in distributed training.

        Args:
            save_path: Path to save NG checkpoint (.pth)
            ng_model: NG model (may be DDP-wrapped)
            ng_optimizer: Optimizer for NG
            ng_scheduler: Scheduler for NG (optional)
            extra: Any extra information to save (optional)
        """
        if self.rank != 0:
            return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        state = {
            "epoch": self.epoch,
            "ng_model_state_dict": (
                ng_model.module.state_dict()
                if isinstance(ng_model, torch.nn.parallel.DistributedDataParallel)
                else ng_model.state_dict()
            ),
            "ng_optimizer_state_dict": ng_optimizer.state_dict(),
            "ng_scheduler_state_dict": ng_scheduler.state_dict()
            if ng_scheduler is not None
            else None,
        }

        if extra is not None:
            state.update(extra)

        torch.save(state, save_path)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load training checkpoint from disk.

        支持：
        - model
        - HL / Adapter（bilevel）
        - NG（对抗）
        - 多 optimizer / scheduler
        """

        map_location = (
            {"cuda:%d" % 0: "cuda:%d" % self.rank}
            if self.distributed
            else self.device
        )
        checkpoint = torch.load(path, map_location=map_location)

        # ---------- model ----------
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # ---------- HL / Adapter ----------
        if (
            hasattr(self, "upper_module")
            and self.upper_module is not None
            and checkpoint.get("hl_state_dict") is not None
        ):
            self.upper_module.load_state_dict(checkpoint["hl_state_dict"])

        if (
            hasattr(self, "lower_module")
            and self.lower_module is not None
            and checkpoint.get("adapter_state_dict") is not None
        ):
            self.lower_module.load_state_dict(checkpoint["adapter_state_dict"])

        # ---------- NG ----------
        if (
            hasattr(self, "ng")
            and self.ng is not None
            and checkpoint.get("ng_state_dict") is not None
        ):
            self.ng.load_state_dict(checkpoint["ng_state_dict"])

        # ---------- optimizers ----------
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and checkpoint.get("optimizer_state_dict") is not None
        ):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if (
            hasattr(self, "upper_optimizer")
            and self.upper_optimizer is not None
            and checkpoint.get("upper_optimizer_state_dict") is not None
        ):
            self.upper_optimizer.load_state_dict(
                checkpoint["upper_optimizer_state_dict"]
            )

        if (
            hasattr(self, "lower_optimizer")
            and self.lower_optimizer is not None
            and checkpoint.get("lower_optimizer_state_dict") is not None
        ):
            self.lower_optimizer.load_state_dict(
                checkpoint["lower_optimizer_state_dict"]
            )

        if (
            hasattr(self, "ng_optimizer")
            and self.ng_optimizer is not None
            and checkpoint.get("ng_optimizer_state_dict") is not None
        ):
            self.ng_optimizer.load_state_dict(
                checkpoint["ng_optimizer_state_dict"]
            )

        # ---------- scheduler ----------
        if (
            hasattr(self, "scheduler")
            and self.scheduler is not None
            and checkpoint.get("scheduler_state_dict") is not None
        ):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # ---------- meta ----------
        self.epoch = checkpoint.get("epoch", 0)
        self.best_metric = checkpoint.get("best_metric", None)

        return checkpoint

