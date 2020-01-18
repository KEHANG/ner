import os
import json
import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence,
                                pad_packed_sequence)

from tagger.models.base import NerBaseModel, NerHeads

class NerLSTM(NerBaseModel):

    def __init__(self,
                 # for embedding_module
                 vocab_size,
                 embedding_dim,
                 # for encoder
                 hidden_dim,
                 lstm_num_layers,
                 bidirectional,
                 # for ner_heads and
                 # tag_to_ix
                 tag_to_ix,
                 **kwargs):

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.bidirectional = bidirectional
        self.tagset_size = len(tag_to_ix)

        embedding_module = nn.Embedding(vocab_size, embedding_dim)

        if bidirectional:
            single_lstm_hidden_dim = hidden_dim // 2
        else:
            single_lstm_hidden_dim = hidden_dim
        encoder = nn.LSTM(embedding_dim,
                          single_lstm_hidden_dim,
                          num_layers=self.lstm_num_layers,
                          bidirectional=bidirectional,
                          batch_first=True)

        ner_heads = NerHeads(hidden_dim, self.tagset_size)

        super(NerLSTM, self).__init__(embedding_module,
                                      encoder,
                                      ner_heads,
                                      tag_to_ix)

    def forward(self, sentences, lengths, tags_batch=None):

        if self.training and tags_batch is None:
            raise ValueError("In training mode, targets should be passed")

        embeds = self.embedding_module(sentences)
        embeds_packed = pack_padded_sequence(embeds,
                                             lengths,
                                             batch_first=True)
        packed_activations, _ = self.encoder(embeds_packed)
        activations, _ = pad_packed_sequence(packed_activations,
                                             batch_first=True)
        outputs = self.ner_heads(activations)

        if self.training:
            loss = nn.NLLLoss()
            return loss(outputs.permute(0, 2, 1), tags_batch)

        return torch.argmax(outputs, dim=2)

    def save(self, output_dir):

        # save model weights
        super(NerLSTM, self).save(output_dir)
        # save model-parameters
        model_params = {
                "vocab_size": self.vocab_size,
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "lstm_num_layers": self.lstm_num_layers,
                "tag_to_ix": self.tag_to_ix,
                "model_type": self.__class__.__name__
        }

        with open(os.path.join(output_dir, 'model_params.json'), 'w') as f_out:
            json.dump(model_params, f_out, indent=3)