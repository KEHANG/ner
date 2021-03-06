import os
import json
import torch
import unittest
from torch import optim
from torch.utils.data import DataLoader

import tagger.models.lstm
from tagger.loader import NERDataset, PadSequence

class TestLSTM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_base = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        model_path = os.path.join(test_base, 'data', 'lstm')
        # pretrained model is used to test
        # inference related functions/methods
        cls.pretrained_model = tagger.models.lstm.NerLSTM.load(model_path)

        with open(os.path.join(model_path, 'word_to_ix.json'), 'r') as f:
            word_to_ix = json.load(f)

        with open(os.path.join(model_path, 'tag_to_ix.json'), 'r') as f:
            tag_to_ix = json.load(f)

        # fresh model is used to test
        # training related functions/methods
        cls.fresh_model = tagger.models.lstm.NerLSTM(
                    vocab_size=14987, embedding_dim=10,
                    hidden_dim=8, lstm_num_layers=1,
                    bidirectional=False, tag_to_ix=tag_to_ix)

        train_data = NERDataset(
                root=os.path.join(test_base, 'data'),
                filename='mini_train.txt',
                word_to_ix=word_to_ix, tag_to_ix=tag_to_ix)

        cls.train_loader = DataLoader(
                    train_data,
                    batch_size=100,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=False,
                    drop_last=True,
                    collate_fn=PadSequence())

    def test_forward(self):

        sentences = torch.tensor([[2173,676,14711,7302,2131,1778,
                                   14591,2340,12260,14041,9]])
        lengths = torch.tensor([11])
        tag_seqs_pred = self.pretrained_model.forward(sentences, lengths)
        tag_seqs = torch.tensor([[4, 4, 6, 4, 4, 4, 4, 4, 4, 4, 4]])
        self.assertEqual(tag_seqs_pred.tolist(), tag_seqs.tolist())

    def test_train_one_epoch(self):

        optimizer = optim.SGD(self.fresh_model.parameters(),
                              lr=0.01, weight_decay=1e-4)
        loss1, _ = self.fresh_model.train_one_epoch(self.train_loader, optimizer)
        loss2, _ = self.fresh_model.train_one_epoch(self.train_loader, optimizer)

        self.assertTrue(loss1 > loss2)