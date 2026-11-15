"""Training, evaluation, checkpointing, and the Fixed-2 pretraining stage."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .algorithms import (
    Fixed2,
    build_algorithm,
    optimizer_parameter_groups,
)
from .config import DotDict, save_config
from .data import EpisodeGenerator, build_dataset, build_generator
from .models import build_model
from .utils import (
    AverageMeter,
    CSVLogger,
    append_jsonl,
    count_parameters,
    describe_device,
    ensure_dir,
    format_seconds,
    resolve_device,
    seed_everything,
)


class Experiment:
    def __init__(self, cfg: DotDict, device: str = "auto"):
        self.cfg = cfg
        self.device = resolve_device(device)
        seed_everything(int(cfg.get("seed", 0)))
        self.output_dir = ensure_dir(cfg.train.get("output_dir", "runs/experiment"))
        save_config(cfg, self.output_dir / "config.yaml")

        self.train_dataset = build_dataset(cfg, "train", train=True)
        self.val_dataset = build_dataset(cfg, "val", train=False)
        self.test_dataset = build_dataset(cfg, "test", train=False)
        self.num_train_classes = len(self.train_dataset.classes)
        self.n_way = int(cfg.dataset.n_way)

        self.train_gen = EpisodeGenerator(
            self.train_dataset,
            self.n_way,
            int(cfg.dataset.n_shot),
            int(cfg.dataset.n_query),
            seed=int(cfg.get("seed", 0)),
        )
        self.val_gen = EpisodeGenerator(
            self.val_dataset,
            self.n_way,
            int(cfg.dataset.n_shot),
            int(cfg.dataset.n_query),
            seed=int(cfg.get("seed", 0)) + 10_000,
        )
        self.test_gen = EpisodeGenerator(
            self.test_dataset,
            self.n_way,
            int(cfg.dataset.n_shot),
            int(cfg.dataset.n_query),
            seed=int(cfg.get("seed", 0)) + 20_000,
        )

        algorithm_name = str(cfg.algorithm.name).lower().replace("_", "-")
        self.is_fixed2 = algorithm_name in {"fixed2", "fixed-2"}
        model_classes = self.num_train_classes if self.is_fixed2 else self.n_way
        self.model = build_model(cfg, num_classes=model_classes).to(self.device)
        self.algorithm = build_algorithm(cfg, self.model, self.device)

        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler = None
        self.start_iter = 1
        self.best_val = -math.inf
        if not self.is_fixed2:
            self._build_optimizer()

        print(f"Device      : {describe_device(self.device)}")
        print(f"Dataset     : {cfg.dataset.name} ({self.num_train_classes} train classes)")
        print(
            f"Model       : Conv4 hidden={cfg.model.hidden} "
            f"params={count_parameters(self.model):,}"
        )
        print(f"Algorithm   : {self.algorithm.name}")
        print(f"Output      : {self.output_dir.resolve()}")

    # ------------------------------------------------------------------
    # optimization
    # ------------------------------------------------------------------
    def _build_optimizer(self) -> None:
        train_cfg = self.cfg.train
        groups = optimizer_parameter_groups(
            self.model,
            self.algorithm,
            meta_lr=float(train_cfg.get("meta_lr", 1e-3)),
            hyper_lr=float(train_cfg.get("hyper_lr", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )
        self.optimizer = torch.optim.Adam(groups)
        schedule = str(train_cfg.get("lr_schedule", "constant")).lower()
        if schedule == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=int(train_cfg.meta_iters)
            )
        elif schedule != "constant":
            raise ValueError(f"Unknown lr_schedule: {schedule}")

    def train(self) -> dict[str, float]:
        if self.is_fixed2:
            return self.train_fixed2()

        assert self.optimizer is not None
        train_cfg = self.cfg.train
        meta_iters = int(train_cfg.meta_iters)
        batch_size = int(train_cfg.get("meta_batch_size", 4))
        eval_every = int(train_cfg.get("eval_every", 500))
        log_every = int(train_cfg.get("log_every", 50))
        save_every = int(train_cfg.get("save_every", 0))
        eval_tasks = int(train_cfg.get("eval_tasks", 100))
        grad_clip = float(train_cfg.get("grad_clip", 10.0))
        fields = [
            "iteration",
            "train_loss",
            "train_acc",
            "train_support_acc",
            "val_loss",
            "val_acc",
            "val_support_acc",
            "mean_lr",
            "l2_lambda",
            "lr",
            "elapsed_sec",
        ]
        logger = CSVLogger(self.output_dir / "log.csv", fields)
        jsonl_path = self.output_dir / "log.jsonl"

        print(f"Start meta-training: {meta_iters} iterations, batch={batch_size}")
        start = time.perf_counter()
        last_val: dict[str, float] = {}

        for iteration in range(self.start_iter, meta_iters + 1):
            episodes = [
                self.train_gen.sample().to(self.device) for _ in range(batch_size)
            ]
            self.optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            metric_meters = {
                "query_loss": AverageMeter(),
                "query_acc": AverageMeter(),
                "support_acc": AverageMeter(),
                "inner_loss": AverageMeter(),
            }
            for episode in episodes:
                loss, step_metrics = self.algorithm.meta_loss(episode)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite meta loss at iteration {iteration}: {loss.item()}"
                    )
                (loss / batch_size).backward()
                total_loss += float(loss.detach().cpu()) / batch_size
                for key, meter in metric_meters.items():
                    meter.update(step_metrics.get(key, 0.0))
                for key in ("mean_lr", "l2_lambda"):
                    if key in step_metrics:
                        metric_meters.setdefault(key, AverageMeter()).update(
                            step_metrics[key]
                        )
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    grad_clip,
                )
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            if iteration % log_every == 0 or iteration == self.start_iter:
                metrics = {
                    key: meter.avg for key, meter in metric_meters.items()
                }
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"[{iteration:6d}/{meta_iters}] "
                    f"loss={total_loss:.4f} "
                    f"query_acc={metrics['query_acc'] * 100:5.1f}% "
                    f"support_acc={metrics['support_acc'] * 100:5.1f}% "
                    f"elapsed={format_seconds(time.perf_counter() - start)}"
                )
                logger.log(
                    {
                        "iteration": iteration,
                        "train_loss": total_loss,
                        "train_acc": metrics["query_acc"],
                        "train_support_acc": metrics["support_acc"],
                        "lr": lr,
                        "mean_lr": metrics.get("mean_lr", ""),
                        "l2_lambda": metrics.get("l2_lambda", ""),
                        "elapsed_sec": round(time.perf_counter() - start, 1),
                    }
                )
                append_jsonl(
                    jsonl_path,
                    {
                        "iteration": iteration,
                        "split": "train",
                        **metrics,
                        "loss": total_loss,
                        "lr": lr,
                    },
                )
                self.save_checkpoint(self.output_dir / "last.pt", iteration)

            if iteration % eval_every == 0 or iteration == meta_iters:
                last_val = self.evaluate("val", n_tasks=eval_tasks)
                print(
                    f"  val: acc={last_val['query_acc'] * 100:.2f}% "
                    f"loss={last_val['query_loss']:.4f} "
                    f"support_acc={last_val['support_acc'] * 100:.2f}%"
                )
                logger.log(
                    {
                        "iteration": iteration,
                        "val_loss": last_val["query_loss"],
                        "val_acc": last_val["query_acc"],
                        "val_support_acc": last_val["support_acc"],
                        "elapsed_sec": round(time.perf_counter() - start, 1),
                    }
                )
                append_jsonl(
                    jsonl_path,
                    {"iteration": iteration, "split": "val", **last_val},
                )
                if last_val["query_acc"] > self.best_val:
                    self.best_val = last_val["query_acc"]
                    self.save_checkpoint(self.output_dir / "best.pt", iteration)
                self.save_checkpoint(self.output_dir / "last.pt", iteration)

            if save_every and iteration % save_every == 0:
                self.save_checkpoint(
                    self.output_dir / f"iter_{iteration:06d}.pt", iteration
                )

        self.save_checkpoint(self.output_dir / "last.pt", meta_iters)
        print(f"Training finished in {format_seconds(time.perf_counter() - start)}")
        return last_val

    # ------------------------------------------------------------------
    # Fixed-2: ordinary supervised pretraining, then 2-step adaptation
    # ------------------------------------------------------------------
    def train_fixed2(self) -> dict[str, float]:
        pretrain_cfg = self.cfg.algorithm.get("pretrain", {})
        epochs = int(pretrain_cfg.get("epochs", 40))
        batch_size = int(pretrain_cfg.get("batch_size", 128))
        lr = float(pretrain_cfg.get("lr", 1e-3))
        weight_decay = float(pretrain_cfg.get("weight_decay", 5e-4))
        workers = int(pretrain_cfg.get("num_workers", 2))
        print(
            f"Fixed-2 pretraining: {epochs} epochs, "
            f"{self.num_train_classes}-way classifier"
        )
        loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs * len(loader))
        )
        start = time.perf_counter()
        for epoch in range(1, epochs + 1):
            meter = AverageMeter()
            correct = 0
            seen = 0
            progress = tqdm(
                loader,
                desc=f"pretrain {epoch}/{epochs}",
                leave=False,
                dynamic_ncols=True,
            )
            for images, targets in progress:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = F.cross_entropy(logits, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()
                meter.update(loss.item(), images.size(0))
                correct += (logits.argmax(1) == targets).sum().item()
                seen += images.size(0)
                progress.set_postfix(
                    loss=f"{meter.avg:.3f}", acc=f"{correct / max(1, seen):.3f}"
                )
            print(
                f"[pretrain {epoch:3d}/{epochs}] loss={meter.avg:.4f} "
                f"acc={correct / max(1, seen) * 100:.2f}% "
                f"elapsed={format_seconds(time.perf_counter() - start)}"
            )

        self.save_checkpoint(
            self.output_dir / "pretrain.pt", iteration=0, include_optimizer=False
        )
        # A fresh N-way head is created for every episode during evaluation.
        self.algorithm.prepare_for_classes(self.n_way)
        val = self.evaluate("val", n_tasks=int(self.cfg.train.get("eval_tasks", 100)))
        print(
            f"Fixed-2 val: acc={val['query_acc'] * 100:.2f}% "
            f"loss={val['query_loss']:.4f}"
        )
        self.save_checkpoint(
            self.output_dir / "last.pt", iteration=0, include_optimizer=False
        )
        if val["query_acc"] > self.best_val:
            self.best_val = val["query_acc"]
            self.save_checkpoint(
                self.output_dir / "best.pt", iteration=0, include_optimizer=False
            )
        return val

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        split: str = "val",
        n_tasks: int = 100,
        progress: bool = False,
    ) -> dict[str, float]:
        if split == "train":
            generator = self.train_gen
        elif split == "val":
            generator = self.val_gen
        elif split == "test":
            generator = self.test_gen
        else:
            raise ValueError(f"Unknown split: {split}")

        meters = {
            "query_loss": AverageMeter(),
            "query_acc": AverageMeter(),
            "support_acc": AverageMeter(),
            "inner_loss": AverageMeter(),
        }
        extra: dict[str, AverageMeter] = {}
        self.model.eval()
        iterator = range(n_tasks)
        if progress:
            iterator = tqdm(iterator, desc=f"eval {split}", leave=False)
        with torch.enable_grad():
            for _ in iterator:
                episode = generator.sample().to(self.device)
                metrics = self.algorithm.evaluate_episode(episode)
                for key in meters:
                    meters[key].update(metrics.get(key, 0.0))
                for key in ("mean_lr", "l2_lambda"):
                    if key in metrics:
                        extra.setdefault(key, AverageMeter()).update(metrics[key])
        self.model.train()
        out = {key: meter.avg for key, meter in meters.items()}
        out.update({key: meter.avg for key, meter in extra.items()})
        out["n_tasks"] = n_tasks
        return out

    # ------------------------------------------------------------------
    # checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(
        self,
        path: str | Path,
        iteration: int,
        include_optimizer: bool = True,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "format_version": 1,
            "config": dict(self.cfg),
            "model": self.model.state_dict(),
            "algorithm": self.algorithm.extra_state_dict(),
            "iteration": int(iteration),
            "best_val": float(self.best_val) if math.isfinite(self.best_val) else -1.0,
            "num_train_classes": self.num_train_classes,
            "n_way": self.n_way,
            "is_fixed2": self.is_fixed2,
        }
        if include_optimizer and self.optimizer is not None:
            payload["optimizer"] = self.optimizer.state_dict()
            if self.scheduler is not None:
                payload["scheduler"] = self.scheduler.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path: str | Path, load_optimizer: bool = True) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        checkpoint_classes = int(payload["model"]["classifier.weight"].shape[0])
        if self.is_fixed2 and checkpoint_classes != self.model.classifier.out_features:
            # ``pretrain.pt`` has the 64-way pretraining head; ``best.pt`` has
            # an N-way head.  Both are valid starting points for Fixed-2, so
            # resize the head to match before loading the state dict.
            self.model.reset_classifier(checkpoint_classes)
        self.model.load_state_dict(payload["model"])
        self.algorithm.load_extra_state_dict(payload.get("algorithm", {}))
        if (
            load_optimizer
            and self.optimizer is not None
            and "optimizer" in payload
        ):
            self.optimizer.load_state_dict(payload["optimizer"])
            if self.scheduler is not None and "scheduler" in payload:
                self.scheduler.load_state_dict(payload["scheduler"])
        self.start_iter = int(payload.get("iteration", 0)) + 1
        self.best_val = float(payload.get("best_val", -1.0))
        if self.is_fixed2:
            self.algorithm.prepare_for_classes(self.n_way)
        print(f"Loaded checkpoint {Path(path).resolve()} (iteration {self.start_iter - 1})")
