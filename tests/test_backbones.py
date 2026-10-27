import importlib.util
import tempfile
import unittest
import torch
from cica.config import ModelConfig, LossConfig
from cica.model import CICA
from cica.losses import objective
from cica.engine import seed_everything


class BackboneTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('transformers'), 'Optional BERT dependencies are not installed')
    def test_standard_bert_parameter_count_on_meta_device(self):
        from transformers import BertConfig
        config = ModelConfig((1, 35, 74), text_mode='bert', hidden_dim=128,
                             bert_config=BertConfig().to_dict())
        with torch.device('meta'):
            model = CICA(config, load_pretrained=False)
        report = model.parameter_inventory()
        self.assertEqual(report['backbone_parameters'], 109482240)
        self.assertEqual(report['total_parameters'], 110348693 + 128 * (35 + 74))
        self.assertEqual(report['projection_input_dims'], [768, 35, 74])

    @unittest.skipUnless(importlib.util.find_spec('transformers'), 'Optional BERT dependencies are not installed')
    def test_local_bert_checkpoint_roundtrip(self):
        from transformers import BertConfig, BertModel
        with tempfile.TemporaryDirectory() as directory:
            spec = BertConfig(vocab_size=97, hidden_size=24, num_hidden_layers=1,
                              num_attention_heads=2, intermediate_size=32,
                              hidden_dropout_prob=0., attention_probs_dropout_prob=0.)
            BertModel(spec).save_pretrained(directory)
            config = ModelConfig((24, 4, 3), text_mode='bert', bert_name=directory,
                                 hidden_dim=12, dropout=0.)
            model = CICA(config).eval()
            streams = [torch.randint(1, 97, (2, 5)), torch.randn(2, 4, 4), torch.randn(2, 3, 3)]
            masks = [torch.ones(x.shape[:2], dtype=torch.bool) for x in streams]
            masks[0][:, -1] = False
            a = model(streams, masks)['yhat']
            rebuilt = CICA(config, load_pretrained=False).eval()
            rebuilt.load_state_dict(model.state_dict())
            torch.testing.assert_close(a, rebuilt(streams, masks)['yhat'])
            changed = [x.clone() for x in streams]
            changed[0][:, -1] = 0
            torch.testing.assert_close(a, model(changed, masks)['yhat'])
            changed[0][:, -1] = 10000  # Invalid vocabulary IDs in padding are ignored.
            torch.testing.assert_close(a, model(changed, masks)['yhat'])
            observed = torch.tensor([[1, 1, 0], [0, 1, 1]], dtype=torch.bool)
            expected = model(streams, masks, observed)['yhat']
            changed[0][1] = 10000
            torch.testing.assert_close(expected, model(changed, masks, observed)['yhat'])
            loss = objective(model(streams, masks, reconstruct=True), torch.tensor([-1., 1.]), config, LossConfig())['total']
            loss.backward()
            gradient = model.text_backbone.embeddings.word_embeddings.weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.abs().sum()), 0.)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_forward_backward(self):
        seed_everything(17)
        config = ModelConfig((5, 4, 3), hidden_dim=12, dropout=0.)
        cpu = CICA(config).eval()
        gpu = CICA(config).cuda().eval()
        gpu.load_state_dict(cpu.state_dict())
        streams = [torch.randn(2, t, d) for t, d in [(5, 5), (4, 4), (3, 3)]]
        masks = [torch.ones(x.shape[:2], dtype=torch.bool) for x in streams]
        a = cpu(streams, masks, reconstruct=True)
        b = gpu([x.cuda() for x in streams], [m.cuda() for m in masks], reconstruct=True)
        torch.testing.assert_close(a['yhat'], b['yhat'].cpu(), rtol=5e-4, atol=5e-4)
        objective(b, torch.tensor([-1., 1.], device='cuda'), config, LossConfig())['total'].backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in gpu.parameters()))


if __name__ == '__main__':
    unittest.main()
