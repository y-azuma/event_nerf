from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class CameraFrame:
    index: int
    frame: int
    time_s: float
    split: str
    file_path: str
    camera_to_target: np.ndarray


class EventScene:
    """Training view of a scene: events, intrinsics and poses, never RGB/mesh."""

    def __init__(self, scene_dir: str | Path):
        self.root = Path(scene_dir).expanduser().resolve()
        payload = json.loads((self.root / "transforms.json").read_text(encoding="utf-8"))
        self.width = int(payload["width"])
        self.height = int(payload["height"])
        self.fx = float(payload["fx"])
        self.fy = float(payload["fy"])
        self.cx = float(payload["cx"])
        self.cy = float(payload["cy"])
        self.frames = [
            CameraFrame(
                index=int(frame["index"]),
                frame=int(frame["frame"]),
                time_s=float(frame["time_s"]),
                split=str(frame["split"]),
                file_path=str(frame["file_path"]),
                camera_to_target=np.asarray(frame["camera_to_target"], dtype=np.float32),
            )
            for frame in payload["frames"]
        ]
        if len(self.frames) < 3:
            raise ValueError("an EventNeRF scene needs at least three poses")
        self.times = np.asarray([frame.time_s for frame in self.frames], dtype=np.float64)
        self.poses = np.stack([frame.camera_to_target for frame in self.frames])
        metadata = json.loads(
            (self.root / "meta" / "metadata.json").read_text(encoding="utf-8")
        )
        sensor = metadata["event_sensor_recommendation"]
        self.positive_threshold = float(
            payload.get("event_threshold_positive", sensor["positive_contrast_threshold"])
        )
        self.negative_threshold = float(
            payload.get("event_threshold_negative", sensor["negative_contrast_threshold"])
        )
        event_path = self.root / payload["events"]
        events = np.load(event_path)
        self.event_x = np.asarray(events["x"], dtype=np.int32)
        self.event_y = np.asarray(events["y"], dtype=np.int32)
        self.event_t = np.asarray(events["t"], dtype=np.float64)
        self.event_p = np.asarray(events["p"], dtype=np.int8)
        self._cumulative: np.ndarray | None = None

    @property
    def pixel_count(self) -> int:
        return self.width * self.height

    def build_cumulative_events(self) -> np.ndarray:
        if self._cumulative is not None:
            return self._cumulative
        delta = np.zeros((len(self.frames), self.pixel_count), dtype=np.float32)
        bins = np.searchsorted(self.times, self.event_t, side="left")
        valid = (
            (bins > 0)
            & (bins < len(self.frames))
            & (self.event_x >= 0)
            & (self.event_x < self.width)
            & (self.event_y >= 0)
            & (self.event_y < self.height)
        )
        bins = bins[valid]
        pixels = self.event_y[valid] * self.width + self.event_x[valid]
        weights = np.where(
            self.event_p[valid] > 0,
            self.positive_threshold,
            -self.negative_threshold,
        ).astype(np.float32)
        np.add.at(delta, (bins, pixels), weights)
        self._cumulative = np.cumsum(delta, axis=0, dtype=np.float32)
        return self._cumulative

    def endpoint_indices(self, split: str) -> np.ndarray:
        indices = [frame.index for frame in self.frames if frame.split == split and frame.index > 0]
        return np.asarray(indices, dtype=np.int32)

    def sample_event_rays(
        self,
        rng: np.random.Generator,
        batch_size: int,
        max_window_frames: int,
        negative_ratio: float,
        endpoint_split: str = "audit_rgb",
    ) -> dict[str, np.ndarray | int]:
        cumulative = self.build_cumulative_events()
        endpoints = self.endpoint_indices(endpoint_split)
        if endpoints.size == 0:
            endpoints = np.arange(1, len(self.frames), dtype=np.int32)
        end = int(rng.choice(endpoints))
        window = int(rng.integers(1, min(max_window_frames, end) + 1))
        start = end - window
        target = cumulative[end] - cumulative[start]
        positive = np.flatnonzero(np.abs(target) > 1e-7)
        negative = np.flatnonzero(np.abs(target) <= 1e-7)
        negative_count = min(batch_size, int(round(batch_size * negative_ratio)))
        positive_count = batch_size - negative_count
        if positive.size == 0:
            positive_count = 0
            negative_count = batch_size
        if negative.size == 0:
            negative_count = 0
            positive_count = batch_size
        chosen = []
        if positive_count:
            chosen.append(rng.choice(positive, positive_count, replace=positive.size < positive_count))
        if negative_count:
            chosen.append(rng.choice(negative, negative_count, replace=negative.size < negative_count))
        pixels = np.concatenate(chosen)
        rng.shuffle(pixels)
        y, x = np.divmod(pixels, self.width)
        jitter = rng.uniform(-0.5, 0.5, size=(pixels.size, 2)).astype(np.float32)
        xy = np.stack([x, y], axis=-1).astype(np.float32) + 0.5 + jitter
        return {
            "xy": xy,
            "target": target[pixels].astype(np.float32),
            "start": start,
            "end": end,
        }

    def rays(
        self,
        xy: np.ndarray | torch.Tensor,
        frame_index: int,
        scene_radius_m: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = torch.as_tensor(xy, dtype=torch.float32, device=device)
        local = torch.stack(
            [
                (pixels[:, 0] - self.cx) / self.fx,
                -(pixels[:, 1] - self.cy) / self.fy,
                -torch.ones_like(pixels[:, 0]),
            ],
            dim=-1,
        )
        pose = torch.as_tensor(self.poses[frame_index], dtype=torch.float32, device=device)
        directions = local @ pose[:3, :3].T
        directions = torch.nn.functional.normalize(directions, dim=-1)
        origins = pose[:3, 3].expand_as(directions) / float(scene_radius_m)
        return origins, directions


def image_grid(width: int, height: int, downsample: int = 1) -> np.ndarray:
    xs = np.arange(0, width, downsample, dtype=np.float32) + 0.5
    ys = np.arange(0, height, downsample, dtype=np.float32) + 0.5
    x, y = np.meshgrid(xs, ys)
    return np.stack([x.ravel(), y.ravel()], axis=-1)
