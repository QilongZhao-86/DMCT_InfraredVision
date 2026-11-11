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
from .blo_strategy import one_step_bilevel_fd
from tqdm import tqdm
from .Adan import Adan
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: torch.utils.data.Dataset,
        *,
        ng: Optional[nn.Module] = None,
        ng_optimizer: Optional[torch.optim.Optimizer] = None,
        de_net: Optional[nn.Module] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        loss_fn: Optional[nn.Module] = None,
        device: Union[str, torch.device] = "cuda",
        batch_size: int = 8,
        num_workers: int = 4,
        distributed: bool = False,
        save_dir: str = "./checkpoints",
        metric_wrapper: Optional[metricWrapper] = metricWrapper(),
        lr_upper: float = 1e-2,
        lr_model: float = 0.01 ,
        epoch: int = 300,
        eta_min: float=1e-6,
        is_ng_train: bool = False,
    ):
        self.distributed=distributed
        self.device = device
        if distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        self.model = model
        self.model = self.model.to(self.device)
        self.scheduler = scheduler
        self.loss_fn = loss_fn if loss_fn is not None else nn.BCELoss()
        self.save_dir = save_dir
        self.metric_wrapper = metric_wrapper
        os.makedirs(self.save_dir, exist_ok=True)
        self.ng=ng
        self.ng = ng.to(self.device) if ng is not None else None
        self.epoch = epoch
        self.best_metric = 0.0
        self.optimizer_model = Adan(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr_model,
            weight_decay=1e-4,
        )
        self.optimizer = self.optimizer_model 
        
        self.optimizer_model_with_ng = None 
        
        if is_ng_train :
            self.param_groups_ng_model = [
                {
                    'params': self.model.parameters(), 
                    'lr': lr_model, 
                    'name': 'detector'
                },
                {
                    'params': self.ng.parameters(), 
                    'lr': 0.0, 
                    'name': 'ng'
                }
            ]
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epoch, eta_min=eta_min
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
        self.upper_module = self.model.image_encoder.hl
        self.upper_params = list(self.upper_module.parameters())
        upper_param_ids = {id(p) for p in self.upper_params}
        self.lower_params = [
            p for p in self.model.parameters()
            if p.requires_grad and id(p) not in upper_param_ids
        ]
        self.upper_optimizer = Adan(self.upper_params,lr=lr_model,weight_decay=1e-4,)
        self.lower_optimizer = Adan(self.lower_params,lr=lr_model*1.25,weight_decay=1e-4,)
        self.scheduler_model=torch.optim.lr_scheduler.CosineAnnealingLR(self.lower_optimizer,T_max=epoch,eta_min=1e-6,)
        self.scheduler_upper=torch.optim.lr_scheduler.CosineAnnealingLR(self.upper_optimizer,T_max=epoch,eta_min=1e-6,)
        if self.rank==0:
            print(
                "model params:",
                sum(1 for _ in self.model.parameters()),
                "upper:",
                len(self.upper_params),
                "lower:",
                len(self.lower_params),
            )
        if self.distributed:
            use_find_unused = True if (self.ng is not None or is_ng_train) else False
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device],
                output_device=self.rank,
                find_unused_parameters=use_find_unused,
            )
        is_ddp = isinstance(self.model, DDP)

        if is_ddp:
            print("当前模型处于 DDP 模式")
        else:
            print("当前模型是单卡训练模式")

        
    def train_one_epoch(self) -> float:
        """
        Train the model for one complete epoch.
        
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.epoch)
        total_loss = 0
        total_samples = 0
        if self.rank == 0:
            manager = enlighten.get_manager()
            pbar = manager.counter(
                total=len(self.train_loader),
                desc=f"[Epoch {self.epoch: 3d}] Training",
                unit="batch",
                color="green",
                leave=False,
            )
        for idx, batch in enumerate(self.train_loader):
            batch_data, batch_masks, _ = batch
            batch_data = batch_data.to(self.device)
            batch_masks = batch_masks.to(self.device)
            pred_logits = self.model(batch_data)
            loss = 0
            for pred_logit in pred_logits:
                loss += self.loss_fn(pred_logit.sigmoid(), batch_masks)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * batch_data.size(0)
            total_samples += batch_data.size(0)
            if self.rank == 0:
                pbar.update()
                pbar.desc = f"[Training] Epoch {self.epoch:3d} loss={total_loss / total_samples: .6f}"
        if self.rank == 0:
            pbar.close()
        avg_loss = total_loss / total_samples
        if self.scheduler is not None:
            self.scheduler.step()
        return avg_loss
    
    
    def activate_model_to_ng_training(self, target_lr=1e-5):
        
        if self.distributed and not isinstance(self.ng, torch.nn.parallel.DistributedDataParallel):
            self.ng = torch.nn.parallel.DistributedDataParallel(
                self.ng, device_ids=[self.device], find_unused_parameters=True
            )
        if self.rank == 0:
            print(f"切换前 Detector LR: {self.optimizer.param_groups[0]['lr']:.8f}")
        actual_last_lr = self.optimizer_model.param_groups[0]['lr']
        current_step = self.scheduler.last_epoch
        t_max = self.scheduler.T_max
        eta_min = self.scheduler.eta_min
        old_detector_initial_lr = self.optimizer_model.param_groups[0].get('initial_lr', 0.01)
        self.optimizer_model_with_ng = Adan(self.param_groups_ng_model, weight_decay=1e-4)
        with torch.no_grad():
            old_state = self.optimizer_model.state
            for group in self.optimizer_model_with_ng.param_groups:
                for p_new in group['params']:
                    if p_new in old_state:
                        self.optimizer_model_with_ng.state[p_new] = {
                            k: v.clone() if isinstance(v, torch.Tensor) else v 
                            for k, v in old_state[p_new].items()
                        }
        self.optimizer = self.optimizer_model_with_ng
        for pg in self.optimizer.param_groups:
            if pg.get('name') == 'detector':
                pg['initial_lr'] = old_detector_initial_lr
            elif pg.get('name') == 'ng':
                pg['initial_lr'] = target_lr
                pg['lr'] = target_lr
                pg['weight_decay'] = 1e-5

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=t_max, 
            eta_min=eta_min,
            last_epoch=current_step
        )
        if abs(self.optimizer.param_groups[0]['lr'] - actual_last_lr) > 1e-5:
            for pg in self.optimizer.param_groups:
                if pg.get('name') == 'detector':
                    pg['lr'] = actual_last_lr

        if self.rank == 0:
            actual_lr = self.optimizer.param_groups[0]['lr']
            print(f"[切换成功] 进度继承至 Epoch {current_step}")
            print(f"切换后 Detector LR: {actual_lr:.8f} ")
            if abs(actual_lr - 0.01) < 1e-5 and current_step > 0:
                print(" 学习率发生了反弹，请检查初始化 lr 设置。")
                

    def train_adv_epoch(self, alpha: float = 0.5, save: bool = False, ng_save_dir: str = "./ng_save"):
        assert self.ng is not None
        SAVE_DIR = ng_save_dir
        total_loss = 0.0
        total_samples = 0
        if save:
            os.makedirs(SAVE_DIR, exist_ok=True)

        self.model.train()
        self.ng.train()
        onesave = True
        for batch_idx, (img, mask, _) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            img = img.to(self.device)
            mask = mask.to(self.device)
            bs = img.size(0)

            img_ng = self.ng(img)

            combined_input = torch.cat([img, img_ng], dim=0)
            
            combined_output = self.model(combined_input) 
 
            pred_clean = []
            pred_ng = []
            for p in combined_output:
                p_c, p_n = torch.split(p, bs, dim=0)
                pred_clean.append(p_c)
                pred_ng.append(p_n)

            loss_clean = sum(self.loss_fn(p.sigmoid(), mask) for p in pred_clean)
            loss_ng = sum(self.loss_fn(p.sigmoid(), mask) for p in pred_ng)
            p_clean_soft = torch.sigmoid(pred_clean[-1]).detach() 
            p_ng_soft = torch.sigmoid(pred_ng[-1])
            loss_consistency = torch.nn.functional.mse_loss(p_ng_soft, p_clean_soft)

            lambda_consist = 0.1 
            loss_all = (alpha * loss_clean) + ((1 - alpha) * loss_ng) + (lambda_consist * loss_consistency)
            
            loss_all.backward()
            self.optimizer.step()

            current_batch_loss = loss_all.item()
            if onesave:
                diff = (img_ng - img).abs().mean()
                print(f"DEBUG: [Joint Batch Mode] [Alpha: {alpha:.2f}] [Intensity: {diff.item():.4f}] [Consist Loss: {loss_consistency.item():.6f}]")
                onesave = False
            if save and self.rank == 0 and batch_idx == 0:
                with torch.no_grad():
                    for i in range(min(bs, 4)): 
                        save_name_orig = os.path.join(SAVE_DIR, f"{i:06d}_orig.png")
                        save_name_deg = os.path.join(SAVE_DIR, f"{i:06d}_deg.png")
                        save_image(img[i], save_name_orig, normalize=True)
                        save_image(img_ng[i], save_name_deg, normalize=True)

            total_loss += current_batch_loss * bs
            total_samples += bs
            
        if self.scheduler is not None:
            self.scheduler.step()

        return total_loss / total_samples
    
    def compute_loss(self,preds, masks):
        return sum(self.loss_fn(p.sigmoid(), masks) for p in preds)
        
    def prepare_bilevel_switch(self):
        steps_per_epoch = len(self.train_loader)  
        
        new_t_max = self.scheduler_model.T_max * steps_per_epoch
        
        self.scheduler_model.T_max = new_t_max
        self.scheduler_upper.T_max = new_t_max
        
        self.model.train()
        
        for p in self.upper_params:
            p.requires_grad = True
        for p in self.lower_params:
            p.requires_grad = True

        self.upper_optimizer.zero_grad(set_to_none=True)
        self.lower_optimizer.zero_grad(set_to_none=True)

        if self.rank == 0:
            print("\n" + "="*40)
            print("SWITCHING TO BILEVEL OPTIMIZATION MODE")
            print(f"Total Epochs: {self.scheduler_model.T_max // steps_per_epoch}")
            print(f"Total Steps:  {self.scheduler_model.T_max}")
            print(f"Lower Initial LR: {self.lower_optimizer.param_groups[0]['lr']:.2e}")
            print(f"Upper Initial LR: {self.upper_optimizer.param_groups[0]['lr']:.2e}")
            print("="*40)

        self.ema_flat_h = None


    def safe_assign_grad(self,param, grad):
        if grad is None:
            return
        if param.grad is None:
            param.grad = torch.autograd.Variable(grad.data)
        else:
            param.grad.data.copy_(grad.data)



    def scale_upper_gradients(self, params, target_norm=1e-3):
        grads = [p.grad for p in params if p.grad is not None]
        if not grads: return
        current_norm = torch.sqrt(sum(g.norm()**2 for g in grads) + 1e-12)
        if current_norm < target_norm:
            scale = target_norm / (current_norm + 1e-12)
            scale = min(scale, 10000.0) 
            for g in grads:
                g.mul_(scale)  

    def train_bilevel_diagnose(self, beta=0.1, delta=1e-8, tau0=5e-2):
        self.model.train()
        stats = {
            "norm_f": [], "norm_h": [], "lambda_k": [],
            "upper_grad_norm": [], "loss_diff": []
        }

        train_iter = iter(self.train_loader)
        for step, batch_tr in enumerate(train_iter):
            img_tr, mask_tr, _ = [b.to(self.device) if torch.is_tensor(b) else b for b in batch_tr]
            if step ==0:
                try: batch_va = next(train_iter)
                except StopIteration: break
                img_va, mask_va, _ = [b.to(self.device) if torch.is_tensor(b) else b for b in batch_va]
            else:
                img_va  = img_tr_old
                mask_va = mask_tr_old

            img_tr_old = img_tr
            mask_tr_old = mask_tr

            x_params = self.upper_params
            y_params = [p for p in self.lower_params if p.requires_grad]

            self.upper_optimizer.zero_grad(set_to_none=True)
            self.lower_optimizer.zero_grad(set_to_none=True)

            g = self.compute_loss(self.model(img_tr), mask_tr)
            v_raw = torch.autograd.grad(g, y_params, allow_unused=True)

            y_active, v_active = [], []
            for p, vi in zip(y_params, v_raw):
                if vi is not None:
                    y_active.append(p)
                    v_active.append(vi.detach())

            if not y_active: continue

            f = self.compute_loss(self.model(img_va), mask_va)
            combined_params = list(x_params) + list(y_active)
            grads_f_all = torch.autograd.grad(f, combined_params, allow_unused=True)

            tau = 2e-4

            with torch.no_grad():
                for p, vi in zip(y_active, v_active): p.add_(tau * vi)
            g_p = self.compute_loss(self.model(img_tr), mask_tr)
            grads_p = torch.autograd.grad(g_p, combined_params, allow_unused=True)

            with torch.no_grad():
                for p, vi in zip(y_active, v_active): p.add_(-2.0 * tau * vi)
            g_m = self.compute_loss(self.model(img_tr), mask_tr)
            grads_m = torch.autograd.grad(g_m, combined_params, allow_unused=True)

            with torch.no_grad():
                for p, vi in zip(y_active, v_active): p.add_(tau * vi)

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

            lambda_val = beta * (n_f / (n_h + delta))

            lambda_val = torch.clamp(lambda_val, min = 0.001, max=1)

            for i, p in enumerate(combined_params):
                p.grad = f_list[i] + lambda_val * h_list[i]

        
            loss_diff = torch.abs(g_p - g_m).item()
            stats["norm_f"].append(n_f.item())
            stats["norm_h"].append(n_h.item())
            stats["lambda_k"].append(lambda_val.item())
            stats["loss_diff"].append(loss_diff)

            self.upper_optimizer.step()
            self.lower_optimizer.step()

            if step % 20 == 0:
                print(f"Step {step} | LossDiff: {loss_diff:.2e} | λ: {lambda_val:.3f} | f_norm: {n_f:.2e}")
        self.scheduler_model.step()
        self.scheduler_upper.step()
        avg_f = sum(stats["norm_f"])/len(stats["norm_f"])
        print(f"\n>>> [Epoch {self.epoch}] BLO Summary: Avg λ={sum(stats['lambda_k'])/len(stats['lambda_k']):.4f}")

        return avg_f

    def scale_upper_gradients(self, params, target_norm=1e-3):
        grads = [p.grad for p in params if p.grad is not None]
        if not grads: return
        current_norm = torch.sqrt(sum(g.norm()**2 for g in grads) + 1e-12)
        if current_norm < target_norm:
            scale = target_norm / (current_norm + 1e-12)
            scale = min(scale, 10000.0)
            for g in grads:
                g.mul_(scale)
                
                
    def _compute_adv_loss_flow(self, img, mask, alpha=0.5, lambda_consist=0.1, save: bool = False, ng_save_dir: str = "./ng_save"):

        bs = img.size(0)
        img_ng = self.ng(img)
        if save and self.rank == 0 :
                with torch.no_grad():
                    for i in range(min(bs, 4)): 
                        save_name_orig = os.path.join(ng_save_dir, f"{i:06d}_orig.png")
                        save_name_deg = os.path.join(ng_save_dir, f"{i:06d}_deg.png")
                        save_image(img[i], save_name_orig, normalize=True)
                        save_image(img_ng[i], save_name_deg, normalize=True)
        
        combined_input = torch.cat([img, img_ng], dim=0)
        combined_output = self.model(combined_input)
        
        if isinstance(combined_output, (list, tuple)):
            pred_clean = [p[:bs] for p in combined_output]
            pred_ng = [p[bs:] for p in combined_output]
            loss_clean = sum(self.loss_fn(p.sigmoid(), mask) for p in pred_clean)
            loss_ng = sum(self.loss_fn(p.sigmoid(), mask) for p in pred_ng)
            p_clean_soft = torch.sigmoid(pred_clean[-1]).detach()
            p_ng_soft = torch.sigmoid(pred_ng[-1])
        else:
            p_c, p_n = torch.split(combined_output, bs, dim=0)
            loss_clean = self.loss_fn(p_c.sigmoid(), mask)
            loss_ng = self.loss_fn(p_n.sigmoid(), mask)
            p_clean_soft = torch.sigmoid(p_c).detach()
            p_ng_soft = torch.sigmoid(p_n)

        loss_consistency = torch.nn.functional.mse_loss(p_ng_soft, p_clean_soft)

        total_loss = (alpha * loss_clean) + ((1 - alpha) * loss_ng) + (lambda_consist * loss_consistency)
        return total_loss
    
    def prepare_combined_switch(self,total_epochs,epoch_now):
        steps_per_epoch = len(self.train_loader)
        self.scheduler_model.T_max = self.scheduler_model.T_max * steps_per_epoch
        self.scheduler_upper.T_max = self.scheduler_upper.T_max * steps_per_epoch
        remaining_epochs = total_epochs - epoch_now
        steps_per_epoch = len(self.train_loader)
        
        new_t_max = remaining_epochs * steps_per_epoch
        
        self.scheduler_model = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.lower_optimizer, 
            T_max=new_t_max, 
            eta_min=1e-6
        )
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = True

        old_state = self.optimizer.state 
        
        def inject_state(new_optimizer):
            with torch.no_grad():
                for group in new_optimizer.param_groups:
                    for p_new in group['params']:
                        if p_new in old_state:
                            # 拷贝 Adan 的所有核心状态：m, v, n
                            new_optimizer.state[p_new] = {
                                k: v.clone() if isinstance(v, torch.Tensor) else v 
                                for k, v in old_state[p_new].items()
                            }
        inject_state(self.upper_optimizer)
        inject_state(self.lower_optimizer)

        actual_last_lr = self.optimizer.param_groups[0]['lr']
        for pg in self.lower_optimizer.param_groups:
            pg['lr'] = actual_last_lr
        for pg in self.upper_optimizer.param_groups:
            pg['lr'] = actual_last_lr * 0.5 

        if self.rank == 0:
            print("="*40)
            print("BILEVEL SWITCH COMPLETED")
            print(f"Momentum states transferred for {len(self.upper_params)} upper and {len(self.lower_params)} lower params.")
            print(f"Lower LR: {self.lower_optimizer.param_groups[0]['lr']:.2e}")
        self.ema_flat_h = None
        
        
        
        
    def train_combined_epoch(self, beta=0.1, delta=1e-8, tau0=5e-2,alpha: float = 0.5, save: bool = False, ng_save_dir: str = "./ng_save"):
        self.model.train()
        self.ng.train()
        
        train_iter = iter(self.train_loader)
        for step, batch_tr in enumerate(train_iter):
            img_tr, mask_tr, _ = [b.to(self.device) if torch.is_tensor(b) else b for b in batch_tr]
            
            if step == 0:
                try: batch_va = next(train_iter)
                except StopIteration: break
                img_va, mask_va, _ = [b.to(self.device) if torch.is_tensor(b) else b for b in batch_va]
            else:
                img_va, mask_va = img_tr_old, mask_tr_old
            img_tr_old, mask_tr_old = img_tr, mask_tr

            x_params = self.upper_params
            y_params = [p for p in self.lower_params if p.requires_grad]
            self.upper_optimizer.zero_grad(set_to_none=True)
            self.lower_optimizer.zero_grad(set_to_none=True)

            g = self._compute_adv_loss_flow(img_tr, mask_tr, alpha=alpha,save=save, ng_save_dir=ng_save_dir)
            v_raw = torch.autograd.grad(g, y_params, allow_unused=True)
            
            y_active, v_active = [], []
            for p, vi in zip(y_params, v_raw):
                if vi is not None:
                    y_active.append(p)
                    v_active.append(vi.detach())
            if not y_active: continue

            f = self.compute_loss(self.model(img_va), mask_va)
            
            combined_params = list(x_params) + list(y_active)
            grads_f_all = torch.autograd.grad(f, combined_params, allow_unused=True)

            tau = 2e-4
            with torch.no_grad():
                for p, vi in zip(y_active, v_active): p.add_(tau * vi)
            g_p = self._compute_adv_loss_flow(img_tr, mask_tr, alpha=alpha,save=False, ng_save_dir=ng_save_dir)
            grads_p = torch.autograd.grad(g_p, combined_params, allow_unused=True)

            with torch.no_grad():
                for p, vi in zip(y_active, v_active): p.add_(-2.0 * tau * vi)
            g_m = self._compute_adv_loss_flow(img_tr, mask_tr, alpha=alpha,save=False, ng_save_dir=ng_save_dir)
            grads_m = torch.autograd.grad(g_m, combined_params, allow_unused=True)

            with torch.no_grad():
                for p, vi in zip(y_active, v_active): p.add_(tau * vi) 
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
            stats = {
                "norm_f": [], "norm_h": [], "lambda_k": [],
                "upper_grad_norm": [], "loss_diff": []
            }
            lambda_val = beta * (n_f / (n_h + delta))

            lambda_val = torch.clamp(lambda_val, min = 0.001, max=1)

            for i, p in enumerate(combined_params):
                p.grad = f_list[i] + lambda_val * h_list[i]

            loss_diff = torch.abs(g_p - g_m).item()
            stats["norm_f"].append(n_f.item())
            stats["norm_h"].append(n_h.item())
            stats["lambda_k"].append(lambda_val.item())
            stats["loss_diff"].append(loss_diff)
            self.upper_optimizer.step()
            self.lower_optimizer.step()
            del f_list, h_list, grads_p, grads_m, grads_f_all, v_active
            lrs = [group['lr'] for group in self.lower_optimizer.param_groups]
            upper_lrs = [group['lr'] for group in self.upper_optimizer.param_groups]

            if step % 20 == 0 and self.rank == 0:
                print(f"Step {step} | Lower LR: {lrs[0]:.8f} | Upper LR: {upper_lrs[0]:.8f}")
                print(f"Step {step} | LossDiff: {loss_diff:.2e} | λ: {lambda_val:.3f} | f_norm: {n_f:.2e}")
                torch.cuda.empty_cache()
        self.scheduler_model.step()
        self.scheduler_upper.step()
        avg_f = sum(stats["norm_f"])/len(stats["norm_f"])
        print(f"\n>>> [Epoch {self.epoch}] BLO Summary: Avg λ={sum(stats['lambda_k'])/len(stats['lambda_k']):.4f}")

        return avg_f
                
        
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

