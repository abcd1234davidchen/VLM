from __future__ import annotations

import unittest

import torch
from torch import nn

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
from vlm.training.xclip_lora import TrainingFeatureAugmentationDataset


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
