import torch
import os
import json
import torch.nn as nn
from seqeval.metrics import f1_score
from torch.nn.utils.rnn import (pack_padded_sequence, 
                                pad_packed_sequence)

class NerLSTM(nn.Module):

    def __init__(self, vocab_size, 
                 tag_to_ix, 
                 embedding_dim, hidden_dim,
                 lstm_num_layers=1, batch_size=2):
        super(NerLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.ix_to_tag = {self.tag_to_ix[tag] : tag for tag in self.tag_to_ix}
        self.tagset_size = len(tag_to_ix)
        self.lstm_num_layers = lstm_num_layers
        self.batch_size = batch_size

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=self.lstm_num_layers,
                            batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, sentences, lengths, tags_batch=None):

        if self.training and tags_batch is None:
            raise ValueError("In training mode, targets should be passed")
        
        embeds = self.word_embeds(sentences)
        embeds_packed = pack_padded_sequence(embeds,
                                             lengths,
                                             batch_first=True)
        packed_activations, _ = self.lstm(embeds_packed)
        activations, _ = pad_packed_sequence(packed_activations, 
                                             batch_first=True)
        outputs = self.hidden2tag(activations)
        outputs = self.log_softmax(outputs)
        
        if self.training:
            loss = nn.NLLLoss()
            return loss(outputs.permute(0, 2, 1), tags_batch)

        return torch.argmax(outputs, dim=2)

    def f1_eval(self, dataloader):

        self.eval()
        all_tag_seqs = []
        all_tag_seqs_pred = []
        for batch in dataloader:
          sentences, lengths, tag_seqs = batch
          tag_seqs_pred= self.forward(sentences, lengths)
          for i, tag_seq_pred in enumerate(tag_seqs_pred):
            length = lengths[i]
            temp_1 =  []
            temp_2 = []
            for j in range(length):
              temp_1.append(self.ix_to_tag[tag_seqs[i][j].item()])
              temp_2.append(self.ix_to_tag[tag_seq_pred[j].item()])

            all_tag_seqs.append(temp_1)
            all_tag_seqs_pred.append(temp_2)

        f1 = f1_score(all_tag_seqs, all_tag_seqs_pred)
        return f1

    def save(self, output_dir):

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        # save model weights
        model_file = os.path.join(output_dir, 'model.pt')
        torch.save(self.state_dict(), model_file)

        # save hyper-parameters
        hyper_params = {
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "lstm_num_layers": self.lstm_num_layers,
                "batch_size": self.batch_size,
                "vocab_size": self.vocab_size,
                "tag_to_ix": self.tag_to_ix,
                "model_type": self.__class__.__name__
        }

        with open(os.path.join(output_dir, 'hyper_params.json'), 'w') as f_out:
            json.dump(hyper_params, f_out, indent=3)