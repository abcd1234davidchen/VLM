"""Contracts for native attention supervision and unchanged RGB inference."""
import unittest

import torch
from transformers import XCLIPConfig, XCLIPModel

from vlm.models.lora import merge_lora, merged_base_state_dict
from vlm.models.xclip_vision_lora import XCLIPVisionClassifier, XCLIPVisionLoRAClassifier
from vlm.native_attention_targets import native_attention_loss, native_pose_distribution, native_patch_distribution
from vlm.training.native_attention import decision


def tiny_base():
    return XCLIPModel(XCLIPConfig(
        text_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1, "num_attention_heads": 4, "vocab_size": 100},
        vision_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
                       "image_size": 224, "patch_size": 32, "num_frames": 2, "mit_hidden_size": 16,
                       "mit_intermediate_size": 32, "mit_num_attention_heads": 4, "mit_num_hidden_layers": 1},
        projection_dim=16, prompt_num_attention_heads=4, attn_implementation="eager"))


class NativeAttentionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_native_output_identity_token_order_and_rgb_export(self):
        torch.manual_seed(42)
        model = XCLIPVisionLoRAClassifier(tiny_base(), num_classes=3, head_dropout=0., upper_blocks=2,
                                         lora_rank=2, lora_alpha=4, lora_dropout=0.).eval()
        x = torch.randn(1, 2, 3, 224, 224)
        with torch.no_grad():
            original = model(x)
            logits, attention = model(x, return_native_attentions=True)
            vision = model.vision_model(x.flatten(0, 1), output_attentions=True, return_dict=True)
        self.assertTrue(torch.equal(logits, original))
        self.assertEqual(attention.shape, (1, 2, 2, 51))
        self.assertTrue(torch.equal(attention[:, :, 0], vision.attentions[0][:, :4, 0].mean(1).reshape(1, 2, 51)))
        self.assertTrue(torch.allclose(attention.sum(-1), torch.ones(1, 2, 2)))
        merge_lora(model)
        restored = XCLIPVisionClassifier(tiny_base(), num_classes=3, head_dropout=0.).eval()
        restored.load_state_dict(merged_base_state_dict(model), strict=True)
        with torch.no_grad():
            self.assertLess(float((restored(x) - original).abs().max()), 1e-5)

    def test_native_loss_reaches_q_and_preserves_frozen_base(self):
        model = XCLIPVisionLoRAClassifier(tiny_base(), num_classes=3, head_dropout=0., upper_blocks=2,
                                         lora_rank=2, lora_alpha=4, lora_dropout=0.).eval()
        frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        logits, attention = model(torch.randn(1, 2, 3, 224, 224), return_native_attentions=True)
        occupancy = torch.zeros(1, 2, 49); occupancy[..., 7] = 1
        mask = torch.ones(1, 2, dtype=torch.bool)
        native_attention_loss(attention, native_pose_distribution(occupancy, mask), mask).backward()
        q = model.vision_model.encoder.layers[-1].self_attn.q_proj.lora_b
        self.assertGreater(float(q.grad.norm()), 0)
        self.assertTrue(torch.isfinite(q.grad).all())
        v = model.vision_model.encoder.layers[-1].self_attn.v_proj.lora_b
        self.assertTrue(v.grad is None or not bool(v.grad.any()))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        optimizer.step()
        for n, p in model.named_parameters():
            if n in frozen:
                self.assertTrue(torch.equal(p, frozen[n]), n)
                self.assertIsNone(p.grad)

    def test_targets_multiple_people_border_and_invalid_frames(self):
        occupied = torch.zeros(1, 3, 49)
        occupied[0, 0, [0, 48]] = 1  # Two people at opposing crop borders.
        occupied[0, 1, 24] = 1
        mask = torch.tensor([[True, True, False]])
        target = native_pose_distribution(occupied, mask)
        self.assertTrue(torch.allclose(target.sum(-1), mask.float()))
        self.assertEqual(float(target[0, 2].sum()), 0)
        self.assertGreater(float(target[0, 0, 0]), .2 / 49)
        self.assertGreater(float(target[0, 0, 48]), .2 / 49)
        with self.assertRaises(ValueError):
            native_pose_distribution(occupied, torch.ones_like(mask))
        attention = torch.rand(1, 3, 2, 51, requires_grad=True)
        zero = native_attention_loss(attention, torch.zeros_like(target), torch.zeros_like(mask))
        zero.backward()
        self.assertEqual(float(zero.detach()), 0)
        self.assertTrue(torch.equal(attention.grad, torch.zeros_like(attention)))

    def test_conditional_distribution_excludes_cls_and_message(self):
        attention = torch.ones(1, 1, 2, 51)
        expected = native_patch_distribution(attention)
        attention[..., 0] = 100; attention[..., 50] = 200
        self.assertTrue(torch.equal(native_patch_distribution(attention), expected))
        self.assertTrue(torch.allclose(expected, torch.full_like(expected, 1 / 49)))

    def test_decision_requires_classification_and_event_gates(self):
        baseline = {"raw_accuracy": .58, "raw_macro_f1": .60, "groups": {"unpaired": {"accuracy": .55}}, "attention_correct": 303}
        current = {**baseline, "raw_accuracy": .59}
        self.assertEqual(decision(current, baseline), "promising")
        self.assertEqual(decision({**current, "attention_correct": 302}, baseline), "inconclusive")
        self.assertEqual(decision({**baseline, "raw_accuracy": .57}, baseline), "no_go")


if __name__ == "__main__":
    unittest.main()
