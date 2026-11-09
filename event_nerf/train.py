from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .config import load_config, seed_everything
from .data import EventScene
from .model import RadianceField
from .rendering import render_rays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EventNeRF from events and known poses")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--config", default="configs/debris_4070.json")
    parser.add_argument("--run-dir", default="runs/1110")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def make_model(config: dict, device: torch.device) -> RadianceField:
    return RadianceField(
        hidden_dim=int(config["hidden_dim"]),
        position_levels=int(config["position_levels"]),
        direction_levels=int(config["direction_levels"]),
    ).to(device)


def forward_batch(
    model: RadianceField,
    scene: EventScene,
    batch: dict,
    config: dict,
    device: torch.device,
    perturb: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    xy = batch["xy"]
    start_o, start_d = scene.rays(
        xy, int(batch["start"]), float(config["scene_radius_m"]), device
    )
    end_o, end_d = scene.rays(
        xy, int(batch["end"]), float(config["scene_radius_m"]), device
    )
    kwargs = {
        "samples_per_ray": int(config["samples_per_ray"]),
        "background_intensity": float(config["background_intensity"]),
        "perturb": perturb,
    }
    start_render = render_rays(model, start_o, start_d, **kwargs)
    end_render = render_rays(model, end_o, end_d, **kwargs)
    gamma = float(config.get("intensity_gamma", 2.2))
    epsilon = float(config.get("tonemap_eps", 0.1))
    prediction = torch.log(end_render["intensity"].pow(gamma) + epsilon) - torch.log(
        start_render["intensity"].pow(gamma) + epsilon
    )
    target = torch.as_tensor(batch["target"], dtype=torch.float32, device=device)
    clip = float(config["event_target_clip"])
    target = target.clamp(-clip, clip)
    prediction = prediction.clamp(-clip, clip)
    extras = {
        "target": target,
        "start_opacity": start_render["opacity"],
        "end_opacity": end_render["opacity"],
        "distortion": 0.5
        * (start_render["distortion"] + end_render["distortion"]),
    }
    return prediction, extras


@torch.no_grad()
def validation_metrics(
    model: RadianceField,
    scene: EventScene,
    batches: list[dict],
    config: dict,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    squared = []
    baseline = []
    for batch in batches:
        prediction, extras = forward_batch(model, scene, batch, config, device, False)
        squared.append(F.mse_loss(prediction, extras["target"]).item())
        baseline.append(extras["target"].square().mean().item())
    return {
        "validation_event_mse": float(np.mean(squared)),
        "validation_event_rmse": float(math.sqrt(np.mean(squared))),
        "zero_prediction_rmse": float(math.sqrt(np.mean(baseline))),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.steps is not None:
        config["steps"] = args.steps
    if args.wandb_mode is not None:
        config["wandb_mode"] = args.wandb_mode
    seed = int(config["seed"])
    seed_everything(seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA is unavailable; training will be slow", flush=True)
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    scene = EventScene(args.scene)
    scene.build_cumulative_events()
    max_window_frames = max(
        1, int(round(len(scene.frames) * float(config["max_window_fraction"])))
    )
    config.update(
        {
            "scene": str(scene.root),
            "run_dir": str(run_dir),
            "device": str(device),
            "max_window_frames": max_window_frames,
            "event_count": int(scene.event_t.size),
            "frame_count": len(scene.frames),
        }
    )
    (run_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    model = make_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    training_rng = np.random.default_rng(seed)
    validation_rng = np.random.default_rng(seed + 1)
    validation_batches = [
        scene.sample_event_rays(
            validation_rng,
            int(config["rays_per_step"]),
            max_window_frames,
            float(config["negative_ray_ratio"]),
            "validation_rgb",
        )
        for _ in range(int(config["validation_batches"]))
    ]

    wandb_run = None
    if config.get("wandb_mode", "disabled") != "disabled":
        try:
            import wandb

            wandb_run = wandb.init(
                project=str(config["wandb_project"]),
                name=f"eventnerf-{run_dir.name}",
                config=config,
                mode=str(config["wandb_mode"]),
                tags=["eventnerf", "debris", "3d-reconstruction", "no-rgb-training"],
            )
        except Exception as error:
            print(f"W&B disabled after initialization error: {error}", flush=True)

    history_path = run_dir / "training_history.csv"
    fields = [
        "step",
        "loss",
        "event_mse",
        "event_rmse",
        "distortion",
        "opacity",
        "negative_ratio",
        "learning_rate",
        "step_time_s",
        "validation_event_mse",
        "validation_event_rmse",
        "zero_prediction_rmse",
    ]
    best_validation = float("inf")
    total_start = time.perf_counter()
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step in range(1, int(config["steps"]) + 1):
            step_start = time.perf_counter()
            progress = (step - 1) / max(1, int(config["steps"]) - 1)
            learning_rate = float(config["learning_rate_final"]) + 0.5 * (
                float(config["learning_rate"]) - float(config["learning_rate_final"])
            ) * (1.0 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            negative_ratio = (
                0.0
                if step <= int(config["negative_ray_warmup_steps"])
                else float(config["negative_ray_ratio"])
            )
            batch = scene.sample_event_rays(
                training_rng,
                int(config["rays_per_step"]),
                max_window_frames,
                negative_ratio,
                "audit_rgb",
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                prediction, extras = forward_batch(model, scene, batch, config, device, True)
                event_mse = F.mse_loss(prediction, extras["target"])
                distortion = extras["distortion"].mean()
                opacity = 0.5 * (
                    extras["start_opacity"].mean() + extras["end_opacity"].mean()
                )
                loss = (
                    event_mse
                    + float(config["distortion_weight"]) * distortion
                    + float(config["opacity_weight"]) * opacity
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            row = {
                "step": step,
                "loss": float(loss.detach()),
                "event_mse": float(event_mse.detach()),
                "event_rmse": float(torch.sqrt(event_mse.detach())),
                "distortion": float(distortion.detach()),
                "opacity": float(opacity.detach()),
                "negative_ratio": negative_ratio,
                "learning_rate": learning_rate,
                "step_time_s": time.perf_counter() - step_start,
                "validation_event_mse": "",
                "validation_event_rmse": "",
                "zero_prediction_rmse": "",
            }
            if step % int(config["validate_every"]) == 0 or step in (1, int(config["steps"])):
                metrics = validation_metrics(model, scene, validation_batches, config, device)
                row.update(metrics)
                if metrics["validation_event_mse"] < best_validation:
                    best_validation = metrics["validation_event_mse"]
                    torch.save(
                        {"model": model.state_dict(), "config": config, "step": step},
                        run_dir / "best.pt",
                    )
            writer.writerow(row)
            handle.flush()
            if wandb_run is not None:
                wandb_run.log({key: value for key, value in row.items() if value != ""}, step=step)
            if step % 50 == 0 or step == 1:
                validation_text = (
                    f" val_rmse={row['validation_event_rmse']:.5f}"
                    if row["validation_event_rmse"] != ""
                    else ""
                )
                print(
                    f"step={step}/{config['steps']} loss={row['loss']:.6f} "
                    f"event_rmse={row['event_rmse']:.5f}{validation_text} "
                    f"time={row['step_time_s']:.3f}s",
                    flush=True,
                )
            if step % int(config["checkpoint_every"]) == 0:
                torch.save(
                    {"model": model.state_dict(), "config": config, "step": step},
                    run_dir / "latest.pt",
                )

    elapsed = time.perf_counter() - total_start
    torch.save(
        {"model": model.state_dict(), "config": config, "step": int(config["steps"])},
        run_dir / "final.pt",
    )
    summary = {
        "training_seconds": elapsed,
        "best_validation_event_mse": best_validation,
        "checkpoint": str(run_dir / "best.pt"),
        "wandb_url": getattr(wandb_run, "url", None) if wandb_run is not None else None,
    }
    (run_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if wandb_run is not None:
        wandb_run.finish()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
