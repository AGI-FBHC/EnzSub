import sys
import unittest
from pathlib import Path

import torch

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from enzsub_seq2topt import (  # noqa: E402
    FeatureAdapter,
    MaskedMultiAttModel,
    Seq2ToptFeatureModel,
)

class MaskedSeq2ToptTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_identity_adapter_preserves_equal_dimensions(self):
        adapter = FeatureAdapter(4, 4)
        values = torch.randn(2, 5, 4)
        self.assertTrue(torch.equal(values, adapter(values)))

    def test_masked_padding_does_not_change_prediction(self):
        model = MaskedMultiAttModel(dim=4, window=1, n_head=2, n_RD=1).eval()
        embedding = torch.randn(1, 4, 6)
        mask = torch.tensor([[False, True, True, True, False, False]])
        prediction = model(embedding, mask)

        extra_padding = torch.randn(1, 4, 3)
        extended_embedding = torch.cat([embedding, extra_padding], dim=-1)
        extended_mask = torch.cat(
            [mask, torch.zeros(1, 3, dtype=torch.bool)], dim=-1
        )
        extended_prediction = model(extended_embedding, extended_mask)
        torch.testing.assert_close(prediction, extended_prediction, rtol=1e-6, atol=1e-6)

    def test_attention_excludes_invalid_positions(self):
        model = MaskedMultiAttModel(dim=4, window=1, n_head=2, n_RD=1).eval()
        embedding = torch.randn(2, 4, 5)
        mask = torch.tensor(
            [
                [False, True, True, False, False],
                [False, True, True, True, False],
            ]
        )
        _, attention = model(embedding, mask, return_attention=True)
        self.assertTrue(torch.equal(attention[~mask], torch.zeros_like(attention[~mask])))
        torch.testing.assert_close(
            attention.sum(dim=1), torch.ones(2), rtol=1e-6, atol=1e-6
        )

    def test_all_invalid_sequence_is_rejected(self):
        model = MaskedMultiAttModel(dim=4, window=1, n_head=2, n_RD=1)
        with self.assertRaisesRegex(ValueError, "at least one valid residue"):
            model(torch.randn(1, 4, 3), torch.zeros(1, 3, dtype=torch.bool))

    def test_task_checkpoint_roundtrip(self):
        first = Seq2ToptFeatureModel(encoder_dim=8, head_dim=4, window=1, n_head=2, n_RD=1)
        second = Seq2ToptFeatureModel(encoder_dim=8, head_dim=4, window=1, n_head=2, n_RD=1)
        second.load_state_dict(first.state_dict(), strict=True)
        features = torch.randn(2, 6, 8)
        mask = torch.ones(2, 6, dtype=torch.bool)
        torch.testing.assert_close(first(features, mask), second(features, mask))

if __name__ == "__main__":
    unittest.main()
