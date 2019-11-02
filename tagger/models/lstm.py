import os
import json
import torch
from tqdm import tqdm
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
                 bidirectional,
                 # for ner_heads
                 tagset_size):

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.bidirectional = bidirectional
        self.tagset_size = tagset_size

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
                                      ner_heads)

    @classmethod
    def load(cls, model_path):
        with open(os.path.join(model_path, 'model_params.json'), 'r') as f:
            model_params = json.load(f)

        model = cls(vocab_size=model_params['vocab_size'],
                    embedding_dim=model_params['embedding_dim'],
                    hidden_dim=model_params['hidden_dim'],
                    lstm_num_layers=model_params['lstm_num_layers'],
                    bidirectional=model_params['bidirectional'],
                    tagset_size=model_params['tagset_size'])

        model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt'),
                                       map_location='cpu'))
        model.eval()

        return model

    def train_one_epoch(self, train_dataloader, optimizer):

        train_loss = 0.0
        nb_tr_steps = 0
        self.train()
        for _, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

            self.zero_grad()
            sentences, lengths, tag_seqs = batch
            # forward pass
            loss = torch.mean(self.forward(sentences, lengths, tag_seqs))

            # backward pass
            loss.backward()
            # track train loss
            train_loss += loss.item()
            nb_tr_steps += 1
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=self.parameters(), max_norm=1.0)
            # update parameters
            optimizer.step()

        return train_loss, nb_tr_steps

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