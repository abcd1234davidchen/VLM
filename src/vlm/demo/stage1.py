"""Stage-1 inference for the live demo: the locked final configuration on one 8-frame window.

Prediction is the mean of A's softmax on center_zoom70 and B's softmax on the person crop, as in
docs/person-crop-test-evaluation-plan.md. The person crop is localized with MediaPipe on the same
8 frames, exactly as in person_crop_localization.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import XCLIPModel, XCLIPProcessor

from vlm.models.xclip_vision_lora import XCLIPVisionClassifier
from vlm.training.person_crop_audit import crop
from vlm.training.person_crop_localization import CONFIDENCE, NUM_POSES, center_zoom70_box, person_square

# Alert events and their per-event thresholds on the validation calibration half
# (alert_false_positive_audit "falling" and "nausea_vomiting" targets): falling uses the 98% recall
# point (no false alerts on the evaluation half), nausea/vomiting the 95% point. Staggering and the body-pain
# poses (A042, A044-A047) are not alerted: their acted NTU versions are not reliable enough.
ALERT_EVENTS = {"A043": "falling", "A048": "nausea/vomiting"}
DEFAULT_THRESHOLDS = {"A043": 0.2010, "A048": 0.1146}

NTU120_NAMES = (
    "drink water", "eat meal/snack", "brushing teeth", "brushing hair", "drop", "pickup", "throw", "sitting down",
    "standing up", "clapping", "reading", "writing", "tear up paper", "wear jacket", "take off jacket",
    "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses", "put on a hat/cap",
    "take off a hat/cap", "cheer up", "hand waving", "kicking something", "reach into pocket", "hopping",
    "jump up", "make a phone call", "playing with phone/tablet", "typing on a keyboard", "pointing to something",
    "taking a selfie", "check time (from watch)", "rub two hands together", "nod head/bow", "shake head",
    "wipe face", "salute", "put the palms together", "cross hands in front", "sneeze/cough", "staggering",
    "falling", "touch head (headache)", "touch chest (stomachache/heart pain)", "touch back (backache)",
    "touch neck (neckache)", "nausea or vomiting", "use a fan", "punching/slapping other person",
    "kicking other person", "pushing other person", "pat on back of other person", "point finger at other person",
    "hugging other person", "giving something to other person", "touch other person's pocket", "handshaking",
    "walking towards each other", "walking apart from each other", "put on headphone", "take off headphone",
    "shoot at the basket", "bounce ball", "tennis bat swing", "juggling table tennis balls", "hush",
    "flick hair", "thumb up", "thumb down", "make ok sign", "make victory sign", "staple book",
    "counting money", "cutting nails", "cutting paper", "snapping fingers", "open bottle", "sniff",
    "squat down", "toss a coin", "fold paper", "ball up paper", "play magic cube", "apply cream on face",
    "apply cream on hand back", "put on bag", "take off bag", "put something into a bag",
    "take something out of a bag", "open a box", "move heavy objects", "shake fist", "throw up cap/hat",
    "hands up (both hands)", "cross arms", "arm circles", "arm swings", "running on the spot", "butt kicks",
    "cross toe touch", "side kick", "yawn", "stretch oneself", "blow nose", "hit other person with something",
    "wield knife towards other person", "knock over other person", "grab other person's stuff",
    "shoot at other person with a gun", "step on foot", "high-five", "cheers and drink",
    "carry something with other person", "take a photo of other person", "follow other person",
    "whisper in other person's ear", "exchange things with other person", "support somebody with hand",
    "rock-paper-scissors",
)


def load_classifier(root: Path, device: torch.device) -> tuple[XCLIPVisionClassifier, XCLIPProcessor]:
    import json
    recipe = json.loads((root / "run_lock.json").read_text(encoding="utf-8"))["recipe"]
    checkpoint = torch.load(root / "final_merged.pt", map_location="cpu", weights_only=False)
    kwargs = {"revision": recipe["model_revision"], "local_files_only": True}
    processor = XCLIPProcessor.from_pretrained(recipe["model_name"], **kwargs)
    model = XCLIPVisionClassifier(XCLIPModel.from_pretrained(recipe["model_name"], **kwargs),
                                  num_classes=recipe["num_classes"], head_dropout=recipe["head_dropout"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), processor


class PoseLocalizer:
    """MediaPipe PoseLandmarker in IMAGE mode with the localization run's settings."""

    def __init__(self, model_path: Path) -> None:
        import mediapipe as mp
        self.mp = mp
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE, num_poses=NUM_POSES,
            min_pose_detection_confidence=CONFIDENCE, min_pose_presence_confidence=CONFIDENCE,
            min_tracking_confidence=CONFIDENCE)
        self.detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def poses(self, frames: np.ndarray) -> list[list[list[list[float]]]]:
        height, width = frames.shape[1:3]
        result = []
        for frame in frames:
            detected = self.detector.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame)))
            result.append([[[p.x * width, p.y * height] for p in landmarks
                            if min(p.visibility, p.presence) >= CONFIDENCE and math.isfinite(p.x) and math.isfinite(p.y)
                            and 0 <= p.x < 1 and 0 <= p.y < 1] for landmarks in detected.pose_landmarks])
        return result


class Stage1:
    def __init__(self, crop_aware: Path, person_trained: Path, pose_model: Path, device: str | None = None) -> None:
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.a, self.processor = load_classifier(crop_aware, self.device)
        self.b, _ = load_classifier(person_trained, self.device)
        self.localizer = PoseLocalizer(pose_model)

    def pixels(self, frames: np.ndarray) -> torch.Tensor:
        return self.processor(images=list(frames), return_tensors="pt")["pixel_values"].to(self.device)

    @torch.inference_mode()
    def predict(self, frames: np.ndarray) -> dict[str, object]:
        """frames: [8, H, W, 3] uint8 RGB, uniformly sampled from the window in time order.

        ``views`` holds the two cropped 8-frame inputs as uint8 arrays, before the XCLIP processor.
        """
        started = time.perf_counter()
        height, width = frames.shape[1:3]
        poses = self.localizer.poses(frames)
        person_box, landmark_box, fallback = person_square(poses, width, height)
        zoom_box = center_zoom70_box(width, height)
        located = time.perf_counter()
        views = {"A_center_zoom70": crop(frames, zoom_box), "B_person_crop": crop(frames, person_box)}
        probability_a = self.a(self.pixels(views["A_center_zoom70"])).float().cpu()[0].softmax(-1)
        probability_b = self.b(self.pixels(views["B_person_crop"])).float().cpu()[0].softmax(-1)
        probability = (probability_a + probability_b) / 2
        finished = time.perf_counter()
        return {
            "events": alert_events(probability), "top": top_classes(probability, 5),
            "models": {name: {"events": alert_events(p), "top": top_classes(p, 5)}
                       for name, p in (("A_center_zoom70", probability_a), ("B_person_crop", probability_b))},
            "views": views,
            "person_box": person_box, "zoom_box": zoom_box, "landmark_box": landmark_box, "fallback": fallback,
            "frame_size": [width, height],
            "poses_middle": poses[len(poses) // 2],
            "timing_ms": {"pose": round(1000 * (located - started)), "classifier": round(1000 * (finished - located))},
        }


def alert_events(probability: torch.Tensor) -> dict[str, float]:
    return {code: float(probability[int(code[1:]) - 1]) for code in ALERT_EVENTS}


def top_classes(probability: torch.Tensor, k: int) -> list[dict[str, object]]:
    top = probability.topk(k)
    return [{"class_id": int(i) + 1, "name": NTU120_NAMES[int(i)], "probability": float(p)}
            for p, i in zip(top.values, top.indices)]
