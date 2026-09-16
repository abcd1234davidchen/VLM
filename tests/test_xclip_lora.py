from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from vlm.models import XCLIPMITClassifier
from vlm.models.lora import (
    LoRALinear,
    audit_trainable_parameters,
    inject_mit_attention_lora,
    load_lora_state_dict,
    lora_state_dict,
    merge_lora,
    merged_base_state_dict,
    unmerge_lora,
)
from vlm.training.xclip_lora import (
    TrainingFeatureAugmentationDataset,
    train_xclip_lora_distillation_epoch,
    train_xclip_lora_epoch,
)


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)


class Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = Attention()


class MIT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([Layer(), Layer()])


class TupleMIT(nn.Module):
    def forward(self, features: torch.Tensor, *, return_dict: bool) -> object:
        pooled = features.mean(dim=1)
        if return_dict:
            return type("Output", (), {"pooler_output": pooled})()
        return (features, pooled)


class XCLIPStub(nn.Module):
    projection_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.mit = TupleMIT()


class XCLIPLoRATest(unittest.TestCase):
    def test_zero_weight_distillation_matches_ce_update(self) -> None:
        torch.manual_seed(12)
        features = torch.randn(6, 4)
        labels = torch.tensor([0, 1, 2, 3, 4, 5])
        teacher = torch.randn(6, 59)
        paired = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.bool)
        sample_ids = tuple(f"sample-{index}" for index in range(6))
        ce_loader = DataLoader(TensorDataset(features, labels), batch_size=6)
        kd_loader = DataLoader(
            list(zip(features, labels, sample_ids, teacher, paired)),
            batch_size=6,
        )
        ce_model = nn.Linear(4, 120)
        kd_model = nn.Linear(4, 120)
        kd_model.load_state_dict(ce_model.state_dict())
        ce_optimizer = torch.optim.SGD(ce_model.parameters(), lr=0.01)
        kd_optimizer = torch.optim.SGD(kd_model.parameters(), lr=0.01)
        train_xclip_lora_epoch(
            ce_model,
            ce_loader,
            ce_optimizer,
            nn.CrossEntropyLoss(label_smoothing=0.05),
            torch.device("cpu"),
            max_grad_norm=100,
            gradient_accumulation_steps=1,
            amp_enabled=False,
            amp_dtype=None,
            epoch=1,
            show_batch_progress=False,
        )
        train_xclip_lora_distillation_epoch(
            kd_model,
            kd_loader,
            kd_optimizer,
            torch.device("cpu"),
            class_indices=torch.arange(59),
            temperature=2,
            kd_weight=0,
            label_smoothing=0.05,
            max_grad_norm=100,
            gradient_accumulation_steps=1,
            amp_enabled=False,
            amp_dtype=None,
        )
        for expected, actual in zip(ce_model.parameters(), kd_model.parameters()):
            self.assertTrue(torch.equal(expected, actual))

    def test_feature_augmentation_does_not_mutate_canonical_tensor(self) -> None:
        class Dataset:
            feature = torch.ones(8, 4)

            def __len__(self) -> int:
                return 1

            def __getitem__(self, index: int):
                return self.feature, torch.tensor(0)

        canonical = Dataset()
        augmented = TrainingFeatureAugmentationDataset(
            canonical,
            frame_dropout=0.5,
            temporal_jitter=1,
            feature_noise_std=0.1,
        )

        features, _ = augmented[0]

        self.assertTrue(torch.equal(canonical.feature, torch.ones(8, 4)))
        self.assertIsNot(features, canonical.feature)

    def test_classifier_requests_structured_mit_output(self) -> None:
        model = XCLIPMITClassifier(
            XCLIPStub(),
            num_classes=2,
            head_dropout=0,
        )

        logits = model(torch.randn(3, 5, 4))

        self.assertEqual(logits.shape, (3, 2))

    def test_injection_targets_only_mit_q_and_v(self) -> None:
        mit = MIT()
        mit.requires_grad_(False)

        targets = inject_mit_attention_lora(
            mit,
            rank=2,
            alpha=4,
            dropout=0,
        )

        self.assertEqual(len(targets), 4)
        for layer in mit.encoder.layers:
            self.assertIsInstance(layer.self_attn.q_proj, LoRALinear)
            self.assertIsInstance(layer.self_attn.v_proj, LoRALinear)
            self.assertIsInstance(layer.self_attn.k_proj, nn.Linear)
            self.assertFalse(layer.self_attn.k_proj.weight.requires_grad)
        audit = audit_trainable_parameters(mit)
        self.assertTrue(audit.names)
        self.assertTrue(
            all(
                name.endswith("lora_a") or name.endswith("lora_b")
                for name in audit.names
            )
        )

    def test_zero_delta_gradient_and_merge_parity(self) -> None:
        base = nn.Linear(4, 3)
        lora = LoRALinear(base, rank=2, alpha=4, dropout=0)
        inputs = torch.randn(5, 4)

        expected = base(inputs)
        initial = lora(inputs)
        self.assertTrue(torch.equal(initial, expected))

        lora(inputs).sum().backward()
        self.assertIsNotNone(lora.lora_b.grad)
        self.assertGreater(lora.lora_b.grad.abs().sum().item(), 0)
        self.assertIsNone(base.weight.grad)

        with torch.no_grad():
            lora.lora_b.normal_()
        unmerged = lora(inputs)
        lora.merge()
        merged = lora(inputs)
        self.assertTrue(torch.allclose(unmerged, merged, atol=1e-6))
        lora.unmerge()
        self.assertTrue(torch.allclose(lora(inputs), unmerged, atol=1e-6))

    def test_adapter_round_trip_and_merged_key_schema(self) -> None:
        source = MIT()
        source.requires_grad_(False)
        inject_mit_attention_lora(
            source,
            rank=2,
            alpha=4,
            dropout=0,
        )
        for parameter in source.parameters():
            if parameter.requires_grad:
                nn.init.normal_(parameter)
        state = lora_state_dict(source)

        target = MIT()
        target.requires_grad_(False)
        inject_mit_attention_lora(
            target,
            rank=2,
            alpha=4,
            dropout=0,
        )
        load_lora_state_dict(target, state)

        for name, value in lora_state_dict(target).items():
            self.assertTrue(torch.equal(value, state[name]))
        merge_lora(target)
        merged = merged_base_state_dict(target)
        self.assertFalse(any("lora_" in name for name in merged))
        self.assertFalse(any(".base." in name for name in merged))
        self.assertIn(
            "encoder.layers.0.self_attn.q_proj.weight",
            merged,
        )
        unmerge_lora(target)


if __name__ == "__main__":
    unittest.main()
