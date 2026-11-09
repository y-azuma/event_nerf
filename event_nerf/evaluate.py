from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.measure import marching_cubes

from .data import EventScene, image_grid
from .metrics import (
    apply_log_affine,
    fit_log_affine,
    geometry_metrics,
    psnr,
    ssim,
)
from .model import RadianceField
from .rendering import render_rays
from .train import forward_batch, make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EventNeRF photometry and 3D geometry")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--run-dir", default="runs/1110")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_gray(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return image[..., 0] * 0.2126 + image[..., 1] * 0.7152 + image[..., 2] * 0.0722


def load_mask(scene_root: Path, frame: int, target: np.ndarray) -> np.ndarray:
    candidates = [
        scene_root / "bbox_masks" / f"frame_{frame:04d}.png",
        scene_root / "bbox_masks" / f"frame_{frame:06d}.png",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return np.asarray(Image.open(candidate).convert("L")) > 0
    # Evaluation-only fallback: the generated background is constant and dark.
    border = np.concatenate([target[0], target[-1], target[:, 0], target[:, -1]])
    background = float(np.median(border))
    return np.abs(target - background) > max(2 / 255, 0.02 * max(background, 1e-3))


@torch.no_grad()
def render_image(
    model: RadianceField,
    scene: EventScene,
    frame_index: int,
    config: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = image_grid(scene.width, scene.height)
    outputs = {"intensity": [], "depth": [], "opacity": []}
    chunk = int(config["render_chunk_rays"])
    model.eval()
    for start in range(0, len(xy), chunk):
        origins, directions = scene.rays(
            xy[start : start + chunk], frame_index, float(config["scene_radius_m"]), device
        )
        rendered = render_rays(
            model,
            origins,
            directions,
            int(config["samples_per_ray"]),
            float(config["background_intensity"]),
            False,
        )
        for key in outputs:
            outputs[key].append(rendered[key].float().cpu().numpy())
    shape = (scene.height, scene.width)
    return tuple(np.concatenate(outputs[key]).reshape(shape) for key in outputs)  # type: ignore[return-value]


def read_ascii_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="ascii").splitlines()
    if lines[:2] != ["ply", "format ascii 1.0"]:
        raise ValueError(f"expected an ASCII PLY: {path}")
    vertex_count = face_count = None
    end = None
    for index, line in enumerate(lines):
        if line.startswith("element vertex "):
            vertex_count = int(line.split()[-1])
        elif line.startswith("element face "):
            face_count = int(line.split()[-1])
        elif line == "end_header":
            end = index + 1
            break
    if vertex_count is None or face_count is None or end is None:
        raise ValueError(f"incomplete PLY header: {path}")
    vertices = np.asarray(
        [[float(value) for value in line.split()[:3]] for line in lines[end : end + vertex_count]],
        dtype=np.float32,
    )
    face_lines = lines[end + vertex_count : end + vertex_count + face_count]
    faces = np.asarray([[int(value) for value in line.split()[1:4]] for line in face_lines], dtype=np.int32)
    return vertices, faces


def write_ascii_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\nend_header\n")
        for vertex in vertices:
            handle.write("{:.7f} {:.7f} {:.7f}\n".format(*vertex))
        for face in faces:
            handle.write("3 {} {} {}\n".format(*face))


def sample_mesh(
    vertices: np.ndarray, faces: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    valid = areas > 1e-12
    triangles, areas = triangles[valid], areas[valid]
    selected = rng.choice(len(triangles), size=count, p=areas / areas.sum())
    u = rng.random(count)
    v = rng.random(count)
    reflect = u + v > 1.0
    u[reflect] = 1.0 - u[reflect]
    v[reflect] = 1.0 - v[reflect]
    tri = triangles[selected]
    return tri[:, 0] + u[:, None] * (tri[:, 1] - tri[:, 0]) + v[:, None] * (
        tri[:, 2] - tri[:, 0]
    )


@torch.no_grad()
def extract_density_mesh(
    model: RadianceField, config: dict, device: torch.device, run_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resolution = int(config["density_grid_resolution"])
    axis = torch.linspace(-1.0, 1.0, resolution)
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
    density_parts = []
    for start in range(0, len(grid), 131072):
        density_parts.append(model.query_density(grid[start : start + 131072].to(device)).cpu())
    density = torch.cat(density_parts).reshape(resolution, resolution, resolution).numpy()
    level = float(config["density_threshold"])
    if not (density.min() < level < density.max()):
        # Fixed threshold is primary; this fallback prevents a meaningless crash
        # and is recorded so it cannot be mistaken for the prespecified metric.
        level = float(np.quantile(density, 0.995))
    vertices, faces, _, _ = marching_cubes(density, level=level)
    vertices = -1.0 + vertices * (2.0 / (resolution - 1))
    vertices *= float(config["scene_radius_m"])
    write_ascii_ply(run_dir / "predicted_mesh.ply", vertices, faces)
    np.save(run_dir / "density_grid.npy", density)
    return vertices.astype(np.float32), faces.astype(np.int32), density


def plot_training(run_dir: Path) -> None:
    with (run_dir / "training_history.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    step = np.asarray([int(row["step"]) for row in rows])
    train = np.asarray([float(row["event_rmse"]) for row in rows])
    val_step = np.asarray([int(row["step"]) for row in rows if row["validation_event_rmse"]])
    val = np.asarray([float(row["validation_event_rmse"]) for row in rows if row["validation_event_rmse"]])
    baseline = np.asarray([float(row["zero_prediction_rmse"]) for row in rows if row["zero_prediction_rmse"]])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(step, train, alpha=0.45, label="train batch")
    axes[0].plot(val_step, val, marker="o", ms=3, label="held endpoint windows")
    axes[0].plot(val_step, baseline, linestyle="--", label="zero-change baseline")
    axes[0].set(xlabel="Optimization step", ylabel="Event log-intensity RMSE", title="Event loss convergence")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    step_time = np.asarray([float(row["step_time_s"]) for row in rows])
    axes[1].plot(step, step_time)
    axes[1].set(xlabel="Optimization step", ylabel="Seconds / step", title="4070 optimization throughput")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(run_dir / "01_training_convergence.png", dpi=180)
    plt.close(fig)


def plot_photometric(
    run_dir: Path,
    views: list[dict],
    examples: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]],
) -> None:
    fig, axes = plt.subplots(len(examples), 3, figsize=(12, 3.4 * len(examples)), squeeze=False)
    for row, (prediction, target, depth, label) in enumerate(examples):
        for col, (image, title, cmap) in enumerate(
            ((target, "GT RGB luminance (test only)", "gray"), (prediction, "EventNeRF corrected", "gray"), (depth, "Predicted depth", "magma"))
        ):
            axes[row, col].imshow(image, cmap=cmap)
            axes[row, col].set_title(f"{label}: {title}")
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(run_dir / "03_novel_view_reconstruction.png", dpi=180)
    plt.close(fig)

    labels = [str(view["frame"]) for view in views]
    full = [view["psnr_db"] for view in views]
    foreground = [view["foreground_psnr_db"] for view in views]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(labels))
    ax.plot(x, full, marker="o", label="full image")
    ax.plot(x, foreground, marker="o", label="target mask")
    ax.set(xlabel="Held-out RGB test frame", ylabel="PSNR [dB]", title="Photometric reconstruction by viewpoint")
    ax.set_xticks(x, labels, rotation=45)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "04_photometric_metrics_by_view.png", dpi=180)
    plt.close(fig)


def plot_geometry(
    run_dir: Path,
    predicted: np.ndarray,
    target: np.ndarray,
    accuracy: np.ndarray,
    completeness: np.ndarray,
    density: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(13, 5.5))
    for index, (points, title, color) in enumerate(
        ((target, "Blender GT surface (evaluation only)", "tab:blue"), (predicted, "EventNeRF density surface", "tab:orange")), start=1
    ):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        subset = points[np.linspace(0, len(points) - 1, min(4000, len(points)), dtype=int)]
        ax.scatter(subset[:, 0], subset[:, 1], subset[:, 2], s=1, alpha=0.45, c=color)
        ax.set_title(title)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
    fig.tight_layout()
    fig.savefig(run_dir / "05_pointcloud_3d_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, max(np.quantile(accuracy, 0.99), np.quantile(completeness, 0.99)), 50)
    ax.hist(accuracy, bins=bins, alpha=0.65, label="prediction → GT (accuracy)")
    ax.hist(completeness, bins=bins, alpha=0.65, label="GT → prediction (completeness)")
    ax.set(xlabel="Nearest surface distance [m]", ylabel="Points", title="Unaligned metric 3D error")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "06_geometry_error_distribution.png", dpi=180)
    plt.close(fig)

    middle = density.shape[0] // 2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, image, title in zip(
        axes,
        (density[middle], density[:, middle], density[:, :, middle]),
        ("x=0", "y=0", "z=0"),
    ):
        displayed = ax.imshow(np.log1p(image), cmap="viridis", origin="lower")
        ax.set_title(f"log density, {title}")
        fig.colorbar(displayed, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(run_dir / "07_density_slices.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else run_dir / "best.pt"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    scene = EventScene(args.scene)
    model = make_model(config, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    start_time = time.perf_counter()

    validation_predictions, validation_targets = [], []
    for frame in scene.frames:
        if frame.split != "validation_rgb":
            continue
        prediction, _, _ = render_image(model, scene, frame.index, config, device)
        target = load_gray(scene.root / frame.file_path)
        validation_predictions.append(prediction)
        validation_targets.append(target)
    slope, intercept = fit_log_affine(validation_predictions, validation_targets)

    view_metrics = []
    examples = []
    lpips_model = None
    try:
        import lpips

        lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    except Exception as error:
        print(f"LPIPS unavailable: {error}", flush=True)
    for frame in scene.frames:
        if frame.split != "test_rgb":
            continue
        raw, depth, opacity = render_image(model, scene, frame.index, config, device)
        prediction = apply_log_affine(raw, slope, intercept)
        target = load_gray(scene.root / frame.file_path)
        mask = load_mask(scene.root, frame.frame, target)
        metrics = {
            "frame": frame.frame,
            "time_s": frame.time_s,
            "psnr_db": psnr(prediction, target),
            "ssim": ssim(prediction, target),
            "foreground_psnr_db": psnr(prediction, target, mask),
            "foreground_mae": float(np.abs(prediction[mask] - target[mask]).mean()) if mask.any() else float("nan"),
            "mean_opacity": float(opacity.mean()),
        }
        if lpips_model is not None:
            a = torch.as_tensor(prediction, device=device)[None, None].repeat(1, 3, 1, 1) * 2 - 1
            b = torch.as_tensor(target, device=device)[None, None].repeat(1, 3, 1, 1) * 2 - 1
            with torch.no_grad():
                metrics["lpips_alex"] = float(lpips_model(a.float(), b.float()))
        view_metrics.append(metrics)
        if len(examples) < 3:
            examples.append((prediction, target, depth * float(config["scene_radius_m"]), f"frame {frame.frame}"))

    vertices, faces, density = extract_density_mesh(model, config, device, run_dir)
    gt_vertices, gt_faces = read_ascii_ply(scene.root / "gt_target_mesh.ply")
    rng = np.random.default_rng(int(config["seed"]) + 2)
    predicted_points = sample_mesh(vertices, faces, int(config["surface_samples"]), rng)
    gt_points = sample_mesh(gt_vertices, gt_faces, int(config["surface_samples"]), rng)
    geometry = geometry_metrics(predicted_points, gt_points, device)
    accuracy = geometry.pop("accuracy_distances")
    completeness = geometry.pop("completeness_distances")
    np.savez_compressed(
        run_dir / "pointclouds_and_errors.npz",
        predicted_points=predicted_points,
        gt_points=gt_points,
        accuracy_m=accuracy,
        completeness_m=completeness,
    )

    mean_photometric = {
        key: float(np.nanmean([view[key] for view in view_metrics if key in view]))
        for key in ("psnr_db", "ssim", "foreground_psnr_db", "foreground_mae", "lpips_alex")
        if any(key in view for view in view_metrics)
    }
    summary = {
        "method": "EventNeRF-compatible grayscale NeRF",
        "checkpoint_step": int(checkpoint["step"]),
        "rgb_training_supervision": False,
        "gt_mesh_training_supervision": False,
        "photometric_calibration": {
            "fit_split": "validation_rgb only",
            "log_slope": slope,
            "log_intercept": intercept,
        },
        "test_rgb_views": len(view_metrics),
        "photometric": mean_photometric,
        "geometry_unaligned_metric_frame": geometry,
        "evaluation_seconds": time.perf_counter() - start_time,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if view_metrics:
        with (run_dir / "photometric_per_view.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(view_metrics[0]))
            writer.writeheader()
            writer.writerows(view_metrics)

    plot_training(run_dir)
    if examples:
        plot_photometric(run_dir, view_metrics, examples)
    plot_geometry(run_dir, predicted_points, gt_points, accuracy, completeness, density)

    # Event-domain test figure uses endpoints reserved from optimizer sampling.
    validation_rng = np.random.default_rng(int(config["seed"]) + 3)
    predicted_events, target_events = [], []
    with torch.no_grad():
        for _ in range(8):
            batch = scene.sample_event_rays(
                validation_rng,
                int(config["rays_per_step"]),
                int(config["max_window_frames"]),
                float(config["negative_ray_ratio"]),
                "validation_rgb",
            )
            prediction, extras = forward_batch(model, scene, batch, config, device, False)
            predicted_events.append(prediction.cpu().numpy())
            target_events.append(extras["target"].cpu().numpy())
    predicted_events = np.concatenate(predicted_events)
    target_events = np.concatenate(target_events)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hexbin(target_events, predicted_events, gridsize=60, mincnt=1, bins="log", cmap="viridis")
    bound = max(0.5, np.quantile(np.abs(target_events), 0.995))
    ax.plot([-bound, bound], [-bound, bound], "r--", label="ideal")
    ax.set(xlabel="Accumulated event target Δlog I", ylabel="Rendered Δlog I", title="Held-endpoint event reconstruction")
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "02_event_reconstruction.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
