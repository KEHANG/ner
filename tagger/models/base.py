import os
import torch
from tqdm import tqdm
import torch.nn as nn
from seqeval.metrics import f1_score

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

        raise NotImplementedError("Subclasses should implement forward()!")

    def forward_on_instance(self, instance):
        """This method is mainly used by model2service."""
        sentence, length = instance
        return self.forward(sentence, length)

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
