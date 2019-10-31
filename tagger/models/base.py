import os
import torch
import torch.nn as nn
from seqeval.metrics import f1_score
from torch.nn.utils.rnn import (pack_padded_sequence,
                                pad_packed_sequence)

class NerBaseModel(nn.Module):

    def __init__(self,
                 embedding_module,
                 encoder,
                 ner_heads):
        super(NerBaseModel, self).__init__()
        self.embedding_module = embedding_module
        self.encoder = encoder
        self.ner_heads = ner_heads

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

    def forward_on_instance(self, instance):
        """This method is mainly used by model2service."""
        sentence, length = instance
        return self.forward(sentence, length)

    def f1_eval(self, dataloader, ix_to_tag):

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
              temp_1.append(ix_to_tag[tag_seqs[i][j].item()])
              temp_2.append(ix_to_tag[tag_seq_pred[j].item()])

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

class NerHeads(nn.Module):
    """Maps the encodings of sentences into tag space"""
    def __init__(self,
                 hidden_dim,
                 tagset_size):
        super(NerHeads, self).__init__()
        
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, activations):

        outputs = self.hidden2tag(activations)
        return self.log_softmax(outputs)
