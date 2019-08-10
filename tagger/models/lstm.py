import torch
import os
import json
import torch.nn as nn

from tagger.models.base import NerBaseModel, NerHeads

class NerLSTM(NerBaseModel):

    def __init__(self,
                 # for embedding_module
                 vocab_size,
                 embedding_dim,
                 # for encoder
                 hidden_dim,
                 lstm_num_layers,
                 # for ner_heads
                 tagset_size):
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.tagset_size = tagset_size

        embedding_module = nn.Embedding(vocab_size, embedding_dim)

        encoder = nn.LSTM(embedding_dim,
                          hidden_dim,
                          num_layers=self.lstm_num_layers,
                          batch_first=True)

        ner_heads = NerHeads(hidden_dim, self.tagset_size)

        super(NerLSTM, self).__init__(embedding_module,
                                      encoder,
                                      ner_heads)

    def save(self, output_dir):

        # save model weights
        super(NerLSTM, self).save(output_dir)
        # save model-parameters
        model_params = {
                "vocab_size": self.vocab_size,
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "lstm_num_layers": self.lstm_num_layers,
                "tagset_size": self.tagset_size,
                "model_type": self.__class__.__name__
        }

        with open(os.path.join(output_dir, 'model_params.json'), 'w') as f_out:
            json.dump(model_params, f_out, indent=3)