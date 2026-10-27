import math
import unittest
from dataclasses import replace
import numpy as np
import torch
from cica.config import ModelConfig, LossConfig
from cica.model import CICA, inference_weights, reconstruction_weights, uncertainty_and_divergence
from cica.losses import objective, reliability_target
from cica.metrics import metrics

torch.set_num_threads(2)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.config = ModelConfig((5, 4, 3), hidden_dim=12, dropout=0)
        self.model = CICA(self.config).eval()
        self.streams = [torch.randn(2, t, d) for t, d in [(5, 5), (4, 4), (3, 3)]]
        self.masks = [torch.ones(x.shape[:2], dtype=torch.bool) for x in self.streams]

    def test_forward_and_optimizer_step(self):
        self.model.train()
        out = self.model(self.streams, self.masks, reconstruct=True)
        losses = objective(out, torch.tensor([-1., 1.]), self.config, LossConfig())
        self.assertEqual(set(losses), {'total', 'task', 'cal', 'rec', 'con', 'div', 'rel'})
        before = self.model.regressor.weight.detach().clone()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        losses['total'].backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, self.model.regressor.weight))
        torch.testing.assert_close(out['alpha'].sum(-1), torch.ones(2))
        torch.testing.assert_close(out['omega'].sum(-1), torch.ones(2))
        self.assertTrue(bool((out['omega'] >= self.config.omega_min).all()))

    def test_attention_ignores_padding(self):
        self.masks[0][:, -2:] = False
        a = self.model(self.streams, self.masks)['encoded']
        changed = [x.clone() for x in self.streams]
        changed[0][:, -2:] = 9999
        b = self.model(changed, self.masks)['encoded']
        torch.testing.assert_close(a, b)

    def test_leave_one_out_has_no_target_input_or_gradient(self):
        shared, private = self.model.decouple(self.model.encode(self.streams, self.masks))
        for target in range(3):
            observed = torch.ones(2, 3, dtype=torch.bool)
            observed[:, target] = False
            a = self.model.reason(shared, private, observed)
            s, p = shared.detach().clone().requires_grad_(), private.detach().clone().requires_grad_()
            grad_s, grad_p = torch.autograd.grad(self.model.reason(s, p, observed)['fused'].sum(), (s, p))
            torch.testing.assert_close(grad_s[:, target], torch.zeros_like(grad_s[:, target]))
            torch.testing.assert_close(grad_p[:, target], torch.zeros_like(grad_p[:, target]))
            changed = [x.clone() for x in self.streams]
            changed[target] = changed[target] + 25
            sh, pr = self.model.decouple(self.model.encode(changed, self.masks))
            b = self.model.reason(sh, pr, observed)
            for field in ['fused', 'intuition', 'alpha', 'reliability', 'conflict', 'gate']:
                torch.testing.assert_close(a[field], b[field])

    def test_padding_nan_is_excluded_before_projection(self):
        self.masks[0][:, -2:] = False
        expected = self.model(self.streams, self.masks)['yhat']
        self.streams[0][:, -2:] = float('nan')
        self.streams[0].requires_grad_()
        actual = self.model(self.streams, self.masks)['yhat']
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertTrue(bool(torch.isfinite(self.streams[0].grad).all()))
        torch.testing.assert_close(self.streams[0].grad[:, -2:], torch.zeros(2, 2, 5))

    def test_unobserved_stream_is_never_encoded(self):
        observed = torch.tensor([[1, 0, 1], [1, 0, 1]], dtype=torch.bool)
        expected = self.model(self.streams, self.masks, observed)['yhat']
        self.streams[1][:] = float('nan')
        def forbidden(*args):
            raise AssertionError('An unobserved modality entered its projection')
        handle = self.model.projections[1].register_forward_pre_hook(forbidden)
        try:
            actual = self.model(self.streams, self.masks, observed)
            torch.testing.assert_close(actual['yhat'], expected)
            torch.testing.assert_close(actual['encoded'][:, 1], torch.zeros(2, 12))
            self.streams[1] = self.streams[1][:, :0]
            self.masks[1] = self.masks[1][:, :0]
            torch.testing.assert_close(self.model(self.streams, self.masks, observed)['yhat'], expected)
        finally:
            handle.remove()

    def test_mixed_missing_inputs_are_excluded_per_sample(self):
        observed = torch.tensor([[1, 1, 0], [0, 0, 1]], dtype=torch.bool)
        expected = self.model(self.streams, self.masks, observed)
        for m, x in enumerate(self.streams):
            x[~observed[:, m]] = float('nan')
        actual = self.model(self.streams, self.masks, observed)
        for key in ['yhat', 'alpha', 'reliability', 'conflict', 'gate']:
            torch.testing.assert_close(actual[key], expected[key])

    def test_single_modality_and_zero_inactive_slots(self):
        for target in range(3):
            observed = torch.zeros(2, 3, dtype=torch.bool)
            observed[:, target] = True
            out = self.model(self.streams, self.masks, observed)
            torch.testing.assert_close(out['alpha'], observed.float())
            torch.testing.assert_close(out['conflict'], torch.zeros(2, 3))
            for key in ['r_stack', 'z_stack']:
                inactive = out[key].reshape(2, 3, -1)[~observed]
                torch.testing.assert_close(inactive, torch.zeros_like(inactive))
            self.assertTrue(torch.isfinite(out['yhat']).all())
            self.assertIsNone(out['omega'])

    def test_mixed_observation_sets(self):
        observed = torch.tensor([[1, 1, 0], [0, 0, 1]], dtype=torch.bool)
        out = self.model(self.streams, self.masks, observed)
        torch.testing.assert_close(out['alpha'].sum(-1), torch.ones(2))
        self.assertEqual(float(out['alpha'][0, 2]), 0)
        self.assertEqual(float(out['alpha'][1, 2]), 1)

    def test_all_missing_is_rejected(self):
        with self.assertRaises(ValueError):
            self.model(self.streams, self.masks, torch.zeros(2, 3, dtype=torch.bool))

    def test_consensus_pair_order_invariance(self):
        shared, private = self.model.decouple(self.model.encode(self.streams, self.masks))
        observed = torch.ones(2, 3, dtype=torch.bool)
        a = self.model.reason(shared, private, observed)['intuition']
        perm = [2, 0, 1]
        b = self.model.reason(shared[:, perm], private[:, perm], observed[:, perm])['intuition']
        torch.testing.assert_close(a, b)

    def test_uniform_is_parameter_matched(self):
        control = CICA(replace(self.config, variant='uniform')).eval()
        control.load_state_dict(self.model.state_dict())
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), sum(p.numel() for p in control.parameters()))
        a, b = self.model(self.streams, self.masks), control(self.streams, self.masks)
        torch.testing.assert_close(b['alpha'], torch.full((2, 3), 1 / 3))
        for field in ['gate', 'reliability', 'conflict', 'omega', 'intuition', 'r_stack', 'z_stack']:
            torch.testing.assert_close(a[field], b[field])

    def test_reliability_target_is_detached(self):
        pred = torch.tensor([[1., -1., 0.]], requires_grad=True)
        target = reliability_target(pred, torch.tensor([0.]), 1.)
        self.assertFalse(target.requires_grad)
        torch.testing.assert_close(target, torch.tensor([[math.exp(-1), math.exp(-1), 1.]]))

    def test_norm_reductions_and_batch_duplication(self):
        out = self.model(self.streams, self.masks, reconstruct=True)
        y = torch.tensor([-1., 1.])
        loss = objective(out, y, self.config, LossConfig())
        manual = sum(((out['shared'][:, i] - out['shared'][:, j]) ** 2).sum(-1).mean()
                     for i in range(3) for j in range(i + 1, 3))
        torch.testing.assert_close(loss['con'], manual)
        repeated = {k: torch.cat([v, v]) if isinstance(v, torch.Tensor) else v for k, v in out.items()}
        duplicate = objective(repeated, y.repeat(2), self.config, LossConfig())
        for k in loss:
            torch.testing.assert_close(loss[k], duplicate[k])

    def test_signed_derivatives_match_closed_form(self):
        r = torch.tensor([[.3, .6, .8]], dtype=torch.float64, requires_grad=True)
        c = torch.tensor([[.8, .3, .1]], dtype=torch.float64, requires_grad=True)
        mask = torch.ones(1, 3, dtype=torch.bool)
        alpha = inference_weights(r, c, mask, .5, .5)
        dr, dc = torch.autograd.grad(alpha[0, 0], (r, c), retain_graph=True)
        a = alpha[0, 0]
        torch.testing.assert_close(dr[0, 0], (1 + .5 * c[0, 0]) * a * (1 - a) / .5)
        torch.testing.assert_close(dc[0, 0], -.5 * (1 - r[0, 0]) * a * (1 - a) / .5)
        omega = reconstruction_weights(r, c, 1., 1., .05)
        derivative = torch.autograd.grad(omega[0, 0], c)[0][0, 0]
        mass = (omega[0, 0] - .05) / .85
        torch.testing.assert_close(derivative, .85 * r[0, 0] * mass * (1 - mass))
        self.assertGreaterEqual(float(derivative), 0)

    def test_stable_entropy_js(self):
        logp = torch.log_softmax(torch.tensor([[[1000., -1000., 0.]]]), -1)
        logq = torch.log_softmax(torch.tensor([[-1000., 1000., 0.]]), -1)
        entropy, js = uncertainty_and_divergence(logp, logq)
        self.assertTrue(torch.isfinite(entropy).all() and torch.isfinite(js).all())
        torch.testing.assert_close(entropy, torch.zeros_like(entropy))
        torch.testing.assert_close(js, torch.ones_like(js))
        e, j = uncertainty_and_divergence(torch.log(torch.full((1, 3, 3), 1 / 3)),
                                         torch.log(torch.full((1, 3), 1 / 3)))
        torch.testing.assert_close(e, torch.ones_like(e))
        torch.testing.assert_close(j, torch.zeros_like(j))

    def test_metric_conventions(self):
        score = metrics([0., -1., 1., .5, 1.5], [-3., -1., 0., .5, 1.5])
        self.assertEqual(score['n_nonzero'], 4)
        self.assertEqual(score['f1'], 100.)
        self.assertEqual(score['acc2'], 100.)
        self.assertEqual(score['acc7'], 60.)
        self.assertIsNone(metrics([0.], [1.])['f1'])

    def test_zero_probability_entropy_js_and_gradients(self):
        for consensus in [[0., -torch.inf, -torch.inf], [-torch.inf, 0., -torch.inf]]:
            lp = torch.tensor([[[0., -torch.inf, -torch.inf]]], dtype=torch.float64, requires_grad=True)
            lq = torch.tensor([consensus], dtype=torch.float64, requires_grad=True)
            entropy, js = uncertainty_and_divergence(lp, lq)
            torch.testing.assert_close(entropy, torch.zeros_like(entropy))
            expected = 0. if consensus[0] == 0. else 1.
            torch.testing.assert_close(js, torch.full_like(js, expected))
            gradients = torch.autograd.grad((entropy + js).sum(), (lp, lq))
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))

    def test_entropy_js_matches_probability_definition(self):
        lp = torch.log_softmax(torch.randn(4, 3, 3, dtype=torch.float64), -1).requires_grad_()
        lq = torch.log_softmax(torch.randn(4, 3, dtype=torch.float64), -1).requires_grad_()
        entropy, js = uncertainty_and_divergence(lp, lq)
        p, q = lp.exp(), lq.exp()[:, None, :]
        mixture = (p + q) / 2
        reference_entropy = -(p * p.log()).sum(-1) / math.log(3)
        reference_js = (p * (p / mixture).log() + q * (q / mixture).log()).sum(-1) / (2 * math.log(2))
        torch.testing.assert_close(entropy, reference_entropy)
        torch.testing.assert_close(js, reference_js)
        actual_grad = torch.autograd.grad((entropy + js).sum(), (lp, lq), retain_graph=True)
        expected_grad = torch.autograd.grad((reference_entropy + reference_js).sum(), (lp, lq))
        for a, b in zip(actual_grad, expected_grad):
            torch.testing.assert_close(a, b)


if __name__ == '__main__':
    unittest.main()
