import json

import numpy as np
import pytest
import torch

from event_nerf.data import EventScene
from event_nerf.metrics import apply_log_affine, fit_log_affine
from event_nerf.model import RadianceField
from event_nerf.rendering import render_rays, sphere_bounds


def test_sphere_bounds_hit_and_miss():
    origins = torch.tensor([[0.0, 0.0, -2.0], [0.0, 0.0, -2.0]])
    directions = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    near, far, valid = sphere_bounds(origins, directions)
    assert valid.tolist() == [True, False]
    assert near[0].item() == pytest.approx(1.0)
    assert far[0].item() == pytest.approx(3.0)


def test_render_is_finite():
    model = RadianceField(hidden_dim=16, position_levels=2, direction_levels=1)
    output = render_rays(
        model,
        torch.tensor([[0.0, 0.0, -2.0]]),
        torch.tensor([[0.0, 0.0, 1.0]]),
        samples_per_ray=8,
        background_intensity=0.001,
        perturb=False,
    )
    assert torch.isfinite(output["intensity"]).all()
    assert 0 < output["intensity"].item() <= 1


def test_log_affine_uses_fixed_validation_map():
    prediction = np.linspace(0.01, 0.9, 100, dtype=np.float32).reshape(10, 10)
    target = np.clip(np.exp(1.2 * np.log(prediction) + 0.1), 0, 1)
    slope, intercept = fit_log_affine([prediction], [target])
    corrected = apply_log_affine(prediction, slope, intercept)
    assert slope == pytest.approx(1.2, rel=1e-2)
    assert np.mean(np.abs(corrected - target)) < 1e-3


def test_event_scene_never_requires_rgb(tmp_path):
    (tmp_path / "meta").mkdir()
    transforms = {
        "width": 4,
        "height": 3,
        "fx": 4.0,
        "fy": 4.0,
        "cx": 2.0,
        "cy": 1.5,
        "events": "events.npz",
        "frames": [
            {
                "index": i,
                "frame": i + 1,
                "time_s": float(i),
                "split": "audit_rgb" if i else "validation_rgb",
                "file_path": f"frames/missing_{i}.png",
                "camera_to_target": np.eye(4).tolist(),
            }
            for i in range(3)
        ],
    }
    (tmp_path / "transforms.json").write_text(json.dumps(transforms))
    (tmp_path / "meta" / "metadata.json").write_text(
        json.dumps(
            {
                "event_sensor_recommendation": {
                    "positive_contrast_threshold": 0.2,
                    "negative_contrast_threshold": 0.2,
                }
            }
        )
    )
    np.savez(tmp_path / "events.npz", x=[1], y=[1], t=[0.5], p=[1])
    scene = EventScene(tmp_path)
    cumulative = scene.build_cumulative_events()
    assert cumulative.shape == (3, 12)
    assert cumulative[1, 5] == pytest.approx(0.2)
