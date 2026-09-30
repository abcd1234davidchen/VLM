"""Gate 3.0 contract for RGB vision adaptation."""

import unittest

import torch
from transformers import XCLIPConfig, XCLIPModel

from vlm.models.lora import lora_state_dict, load_lora_state_dict, merge_lora, merged_base_state_dict
from vlm.models.xclip_vision_lora import XCLIPAttentionClassifier, XCLIPAttentionLoRAClassifier, XCLIPVisionLoRAClassifier


def tiny_xclip() -> XCLIPModel:
    config = XCLIPConfig(
        text_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1, "num_attention_heads": 4, "vocab_size": 100},
        vision_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "image_size": 32, "patch_size": 16, "num_frames": 2, "mit_hidden_size": 16, "mit_intermediate_size": 32, "mit_num_attention_heads": 4, "mit_num_hidden_layers": 1},
        projection_dim=16,
        prompt_num_attention_heads=4,
    )
    return XCLIPModel(config)


class XCLIPVisionLoRATest(unittest.TestCase):
    def test_attention_only_training_preserves_complete_baseline(self) -> None:
        from vlm.training.spatial_attention import configure_adaptation
        model = XCLIPAttentionLoRAClassifier(tiny_xclip(), num_classes=3, head_dropout=0.25,
                                            upper_blocks=1, lora_rank=2, lora_alpha=4, lora_dropout=0.05)
        configure_adaptation(model, "attention-only")
        trainable = {name for name, value in model.named_parameters() if value.requires_grad}
        self.assertEqual(trainable, {"pooling_alpha", "spatial_head.weight", "spatial_head.bias"})
        self.assertFalse(model.training)
        frozen = {name: value.clone() for name, value in model.state_dict().items() if name not in trainable}
        optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad], lr=0.001)
        logits, spatial = model(torch.randn(1, 2, 3, 32, 32), return_spatial=True)
        loss = logits.square().sum() - spatial.log_softmax(-1)[..., 0].mean()
        loss.backward()
        self.assertGreater(float(model.pooling_alpha.grad.abs()), 0)
        self.assertGreater(float(model.spatial_head.weight.grad.abs().sum()), 0)
        optimizer.step()
        for name, original in frozen.items():
            self.assertTrue(torch.equal(model.state_dict()[name], original), name)

    def test_attention_identity_coupling_and_rgb_only_merged_reload(self) -> None:
        torch.manual_seed(42)
        base = tiny_xclip()
        state = {name: value.clone() for name, value in base.state_dict().items()}
        kwargs = dict(num_classes=3, head_dropout=0.0, upper_blocks=1,
                      lora_rank=2, lora_alpha=4.0, lora_dropout=0.0)
        baseline = XCLIPVisionLoRAClassifier(base, **kwargs).eval()
        fresh = tiny_xclip()
        fresh.load_state_dict(state)
        model = XCLIPAttentionLoRAClassifier(fresh, **kwargs).eval()
        load_lora_state_dict(model, lora_state_dict(baseline))
        model.classifier.load_state_dict(baseline.classifier.state_dict())
        pixels = torch.randn(1, 2, 3, 32, 32)
        self.assertTrue(torch.equal(baseline(pixels), model(pixels)))
        model(pixels).square().sum().backward()
        self.assertGreater(float(model.pooling_alpha.grad.abs()), 0)
        self.assertEqual(float(model.spatial_head.weight.grad.abs().sum()), 0)
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.pooling_alpha.fill_(0.2)
        logits, attention = model(pixels, return_spatial=True)
        logits.square().sum().backward()
        self.assertGreater(float(model.spatial_head.weight.grad.abs().sum()), 0)
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                            for name, parameter in model.named_parameters()
                            if name.startswith("vision_model.") and name.endswith("lora_b")))
        self.assertTrue(all(not parameter.requires_grad for name, parameter in model.named_parameters()
                            if name.startswith("vision_model.") and not name.endswith(("lora_a", "lora_b"))))
        merge_lora(model)
        merged_state = merged_base_state_dict(model)
        merged_base = tiny_xclip()
        restored = XCLIPAttentionClassifier(merged_base, num_classes=3, head_dropout=0.0).eval()
        restored.load_state_dict(merged_state, strict=True)
        with torch.no_grad():
            restored_logits, restored_attention = restored(pixels, return_spatial=True)
        self.assertLess(float((restored_logits - logits).detach().abs().max()), 1e-5)
        self.assertLess(float((restored_attention - attention).detach().abs().max()), 1e-5)

    def test_rgb_forward_gradients_merge_and_reload(self) -> None:
        torch.manual_seed(42)
        base = tiny_xclip()
        base_state = {name: value.clone() for name, value in base.state_dict().items()}
        model = XCLIPVisionLoRAClassifier(
            base, num_classes=3, head_dropout=0.0, upper_blocks=1,
            lora_rank=2, lora_alpha=4.0, lora_dropout=0.0,
        )
        self.assertEqual(
            model.target_modules[:2],
            ["vision_model.encoder.layers.1.self_attn.q_proj", "vision_model.encoder.layers.1.self_attn.v_proj"],
        )
        pixels = torch.randn(1, 2, 3, 32, 32)
        model.train()
        logits = model(pixels)
        self.assertEqual(tuple(logits.shape), (1, 3))
        logits.square().sum().backward()
        trainable = model.trainable_parameter_audit().names
        self.assertTrue(any(name.startswith("vision_model.") for name in trainable))
        self.assertTrue(any(name.startswith("mit.") for name in trainable))
        self.assertTrue(all(name.endswith(("lora_a", "lora_b")) or name.startswith("classifier.") for name in trainable))
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for name, parameter in model.named_parameters() if name.startswith("vision_model.") and parameter.requires_grad))

        adapter = lora_state_dict(model)
        head = {name: value.clone() for name, value in model.classifier.state_dict().items()}
        model.eval()
        with torch.no_grad():
            reference = model(pixels)
            merge_lora(model)
            merged = model(pixels)
        self.assertLess((reference - merged).abs().max().item(), 1e-5)
        self.assertIn("vision_model.encoder.layers.1.self_attn.q_proj.weight", merged_base_state_dict(model))

        fresh_base = tiny_xclip()
        fresh_base.load_state_dict(base_state, strict=True)
        restored = XCLIPVisionLoRAClassifier(
            fresh_base, num_classes=3, head_dropout=0.0, upper_blocks=1,
            lora_rank=2, lora_alpha=4.0, lora_dropout=0.0,
        )
        load_lora_state_dict(restored, adapter)
        restored.classifier.load_state_dict(head, strict=True)
        restored.eval()
        with torch.no_grad():
            reloaded = restored(pixels)
        self.assertLess((reference - reloaded).abs().max().item(), 1e-6)


if __name__ == "__main__":
    unittest.main()
