"""Native attention indexing, gradient isolation and bounded review contracts."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import XCLIPConfig, XCLIPModel

from vlm.models.lora import merge_lora, merged_base_state_dict
from vlm.models.patch16_native import XCLIPNativeLoRAClassifier
from vlm.models.xclip_vision_lora import XCLIPVisionClassifier
from vlm.patch16_native_targets import (NativeTargetCache, native_attention_loss,
                                       native_budget_decision, native_patch_distribution,
                                       native_pose_distribution)


def tiny_xclip():
    config = XCLIPConfig(
        text_config={'hidden_size': 16, 'intermediate_size': 32, 'num_hidden_layers': 1,
                     'num_attention_heads': 4, 'vocab_size': 100},
        vision_config={'hidden_size': 16, 'intermediate_size': 32, 'num_hidden_layers': 2,
                       'num_attention_heads': 4, 'image_size': 224, 'patch_size': 16,
                       'num_frames': 2, 'mit_hidden_size': 8, 'mit_intermediate_size': 16,
                       'mit_num_attention_heads': 4, 'mit_num_hidden_layers': 1},
        projection_dim=8, prompt_num_attention_heads=4)
    return XCLIPModel(config)


class NativeContracts(unittest.TestCase):
    def test_patch_slicing_excludes_cls_and_message_including_gradients(self):
        for count in (4, 196):
            attention = torch.ones(1, 2, 2, count + 2, requires_grad=True)
            target = torch.zeros(1, 2, count); target[..., 0] = 1
            loss = native_attention_loss(attention, target, torch.ones(1, 2, dtype=torch.bool))
            loss.backward()
            self.assertEqual(native_patch_distribution(attention).shape[-1], count)
            self.assertTrue(torch.equal(attention.grad[..., 0], torch.zeros_like(attention.grad[..., 0])))
            self.assertTrue(torch.equal(attention.grad[..., -1], torch.zeros_like(attention.grad[..., -1])))
            self.assertGreater(float(attention.grad[..., 1:-1].abs().sum()), 0)
        with self.assertRaises(ValueError):
            native_patch_distribution(torch.zeros(1, 2, 2, 198))

    def test_gaussian_floor_normalization_and_all_invalid(self):
        occupancy = torch.zeros(1, 2, 196); occupancy[0, 0, 0] = 1
        mask = torch.tensor([[True, False]])
        target = native_pose_distribution(occupancy, mask)
        self.assertEqual(target.dtype, torch.float32)
        self.assertTrue(torch.allclose(target.sum(-1), mask.float(), atol=1e-6))
        self.assertGreater(float(target[0, 0, 0]), float(target[0, 0, 1]))
        self.assertGreater(float(target[0, 0, 1]), float(target[0, 0, 3]))
        self.assertAlmostEqual(float(target[0, 0, -1]), .2 / 196, places=7)
        attention = torch.ones(1, 2, 2, 198, requires_grad=True)
        empty = torch.zeros_like(mask)
        loss = native_attention_loss(attention, torch.zeros_like(target), empty)
        self.assertEqual(float(loss.detach()), 0)
        loss.backward(); self.assertTrue(torch.equal(attention.grad, torch.zeros_like(attention)))
        with self.assertRaises(ValueError):
            native_pose_distribution(torch.zeros_like(occupancy), torch.ones_like(mask))

    def test_opt_in_identity_q_gradient_and_rgb_only_merged_reload(self):
        torch.manual_seed(42)
        model = XCLIPNativeLoRAClassifier(tiny_xclip(), num_classes=3, head_dropout=0,
                                         upper_blocks=2, lora_rank=2, lora_alpha=4, lora_dropout=0).eval()
        pixels = torch.randn(1, 2, 3, 224, 224)
        logits = model(pixels)
        enabled, attention = model(pixels, return_native_attentions=True)
        self.assertTrue(torch.equal(logits, enabled))
        self.assertEqual(attention.shape, (1, 2, 2, 198))
        self.assertTrue(torch.allclose(attention.sum(-1), torch.ones(1, 2, 2), atol=1e-5))
        target = torch.zeros(1, 2, 196); target[..., 50] = 1
        native_attention_loss(attention, target, torch.ones(1, 2, dtype=torch.bool)).backward()
        params = dict(model.named_parameters())
        q = params['vision_model.encoder.layers.1.self_attn.q_proj.lora_b']
        v = params['vision_model.encoder.layers.1.self_attn.v_proj.lora_b']
        self.assertGreater(float(q.grad.abs().sum()), 0)
        self.assertTrue(v.grad is None or float(v.grad.abs().sum()) == 0)
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
        self.assertTrue(all(p.grad is None for n, p in params.items() if n.startswith('classifier.')))
        torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001).step()
        before = model(pixels).detach()
        merge_lora(model)
        clean = XCLIPVisionClassifier(tiny_xclip(), num_classes=3, head_dropout=0).eval()
        clean.load_state_dict(merged_base_state_dict(model), strict=True)
        self.assertTrue(torch.allclose(before, clean(pixels), atol=1e-5, rtol=1e-5))
        with self.assertRaises(ValueError):
            model(pixels, return_features=True, return_native_attentions=True)

    def test_derived_cache_rejects_corrupted_target(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = NativeTargetCache(Path(directory) / 'spatial', Path(directory) / 'native', {})
            occupancy = torch.zeros(8, 196); occupancy[:, 80] = 1
            source = {'target': occupancy, 'valid': torch.ones(8, dtype=torch.bool),
                      'fingerprint': {'source': 'validated-spatial'}}
            cache.spatial.load = lambda row: copy.deepcopy(source)
            row = {'sample_id': 'sample'}
            first = cache.load(row);second = cache.load(row)
            self.assertTrue(torch.equal(first['target'], second['target']))
            path = next(cache.root.rglob('*.pt'))
            broken = torch.load(path, weights_only=True); broken['target'][0, 0] += .1
            torch.save(broken, path)
            with self.assertRaises(ValueError):cache.load(row)

    def test_two_epoch_budget_never_extends_automatically(self):
        self.assertEqual(native_budget_decision(1, {}), 'continue')
        self.assertEqual(native_budget_decision(2, {}), 'two_epoch_review_no_candidate')
        self.assertEqual(native_budget_decision(2, {'whole': {'epoch': 1}}), 'two_epoch_review_promising')


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
