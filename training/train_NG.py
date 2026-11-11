import os
from typing import Optional, Dict, Any, Tuple, Union
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from loguru import logger
import enlighten
import torch.nn.functional as F

# Import metricWrapper for type hints
from .metrics_config import metricWrapper
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataset: torch.utils.data.Dataset,
        *,
        ng: Optional[nn.Module] = None,
        ng_optimizer: Optional[torch.optim.Optimizer] = None,
        de_net: Optional[nn.Module] = None,
        ng_start_epoch: int = 0,
        ng_end_epoch: int = 1_000_000,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        loss_fn: Optional[nn.Module] = None,
        device: Union[str, torch.device] = "cuda",
        batch_size: int = 8,
        num_workers: int = 4,
        distributed: bool = False,
        save_dir: str = "./checkpoints",
        metric_wrapper: Optional[metricWrapper] = metricWrapper(),
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn if loss_fn is not None else nn.BCELoss()
        self.device = device
        self.save_dir = save_dir
        self.metric_wrapper = metricWrapper()
        os.makedirs(self.save_dir, exist_ok=True)

        self.distributed = distributed
        self.rank = dist.get_rank() if distributed else 0
        self.world_size = dist.get_world_size() if distributed else 1
        self.epoch = 0
        self.best_metric = None

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

        self.model = self.model.to(self.device)
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device] if torch.cuda.is_available() else None,
                output_device=self.rank,
                find_unused_parameters=True,
            )

        self.ng = ng.to(self.device) if ng is not None else None
        self.ng_optimizer = ng_optimizer
        self.de_net = de_net.to(self.device) if de_net is not None else None

        if self.de_net is not None:
            self.de_net.eval()
            for p in self.de_net.parameters():
                p.requires_grad_(False)

        self.ng_start_epoch = ng_start_epoch
        self.ng_end_epoch = ng_end_epoch
    def adversarial_ng_step(
        *,
        model: nn.Module,
        ng: nn.Module,
        de_net: Optional[nn.Module],
        img: torch.Tensor,
        mask: torch.Tensor,
        loss_fn: nn.Module,
        ng_optimizer: torch.optim.Optimizer,
        ng_steps: int = 1,
        w_de: float = 1.0,
        w_struct: float = 0.5,
        w_task: float = 1.0,
    ):
        """
        Inner maximization: update NG only, model is frozen.
        """
        model.eval()    
        ng.train()

        for _ in range(ng_steps):
            lq = ng(img)
            with torch.no_grad():
                preds = model(lq)
            task_loss = sum(loss_fn(p.sigmoid(), mask) for p in preds)
            struct_loss = F.l1_loss(lq, img)
            if de_net is not None:
                with torch.no_grad():
                    deg_score = de_net(lq)
                de_loss = -deg_score.mean()
            else:
                de_loss = 0.0
            ng_loss = (
                w_de * de_loss
                + w_struct * struct_loss
                - w_task * task_loss
            )

            ng_optimizer.zero_grad()
            ng_loss.backward()
            ng_optimizer.step()
        model.train()
    
    ## 双层训练函数
    def train_bilevel_epoch(
        self,
        *,
        w_de: float = 1.0,
        w_struct: float = 0.5,
        w_task: float = 1.0,
    ) -> Dict[str, float]:
        """
        One epoch of bilevel training:
        - update model (detector)
        - update NG (degradation generator)
        """

        assert self.ng is not None, "NG model is required"
        assert self.ng_optimizer is not None, "NG optimizer is required"
        assert self.de_net is not None, "DE network is required"
        if self.rank == 0:
            print("you are bi-level training")
        self.model.train()
        self.ng.train()

        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.epoch)
            
        stat = {
            "model_loss": 0.0,
            "ng_loss": 0.0,
        }
        total_samples = 0

        for batch in self.train_loader:
            img, mask, _ = batch
            img = img.to(self.device)
            mask = mask.to(self.device)
            bs = img.size(0)
            total_samples += bs

            # ===== shared forward =====
            lq = self.ng(img)

            # ===== 1) MODEL STEP =====
            with torch.no_grad():          # 断开 NG ← MODEL
                lq_detached = lq.detach()

            preds = self.model(lq_detached)
            model_loss = sum(self.loss_fn(p.sigmoid(), mask) for p in preds)

            self.optimizer.zero_grad()
            model_loss.backward()
            self.optimizer.step()

            # ===== 2) NG STEP =====
            if self.ng_start_epoch <= self.epoch <= self.ng_end_epoch:
                self.model.eval()

                # DE loss
                with torch.no_grad():
                    deg_score = self.de_net(lq)
                de_loss = -deg_score.mean()

                # structure loss
                struct_loss = F.l1_loss(lq, img)

                # task loss (stop-grad through model)
                preds_lq = self.model(lq.detach())   # 不更新 model
                task_loss = sum(self.loss_fn(p.sigmoid(), mask) for p in preds_lq)


                ng_loss = (
                    w_de * de_loss
                    + w_struct * struct_loss
                    - w_task * task_loss
                )

                self.ng_optimizer.zero_grad()
                ng_loss.backward()
                self.ng_optimizer.step()

                self.model.train()


        # -------- epoch stats --------
        for k in stat:
            stat[k] /= total_samples

        if self.scheduler is not None:
            self.scheduler.step()

        return stat
    
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
        """
        Save training checkpoint to disk.
        
        Only rank 0 process saves checkpoints in distributed training.
        
        Args:
            save_path: Path to save the checkpoint
            extra: Additional data to include in checkpoint (optional)
        """
        if self.rank != 0:
            return
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        state = {
            "epoch": self.epoch,
            "model_state_dict": self.model.module.state_dict()
            if self.distributed
            else self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "best_metric": self.best_metric,
        }
        if extra:
            state.update(extra)
        torch.save(state, save_path)
        
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
        
        Args:
            path: Path to the checkpoint file
            
        Returns:
            dict: Loaded checkpoint dictionary
        """
        map_location = (
            {"cuda:%d" % 0: "cuda:%d" % self.rank} if self.distributed else self.device
        )
        checkpoint = torch.load(path, map_location=map_location)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint.get("epoch", 0)
        self.best_metric = checkpoint.get("best_metric", None)
        return checkpoint
