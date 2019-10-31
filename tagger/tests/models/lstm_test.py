import os
import torch
import unittest

import tagger.models.lstm

class TestLSTM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_base = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        model_path = os.path.join(test_base, 'data', 'lstm')
        cls.model = tagger.models.lstm.NerLSTM.load(model_path)

    def test_forward(self):

        sentences = torch.tensor([[2173,676,14711,7302,2131,1778,
                                   14591,2340,12260,14041,9]])
        lengths = torch.tensor([11])
        tag_seqs_pred = self.model.forward(sentences, lengths)
        tag_seqs = torch.tensor([[4, 4, 6, 4, 4, 4, 4, 4, 4, 4, 4]])
        self.assertEqual(tag_seqs_pred.tolist(), tag_seqs.tolist())

