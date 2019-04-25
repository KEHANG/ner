import torch
import torch.nn as nn
torch.manual_seed(1)

from tagger.constant import START_TAG, STOP_TAG

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim,
                 lstm_num_layers=1, batch_size=2):
        super(BiLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.lstm_num_layers = lstm_num_layers
        self.batch_size = batch_size

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.hidden = self.init_hidden()

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=self.lstm_num_layers, bidirectional=True,
                            batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.log_softmax = nn.LogSoftmax(dim=2)

    def init_hidden(self):
        return (torch.randn(self.lstm_num_layers*2, self.batch_size, self.hidden_dim // 2),
                torch.randn(self.lstm_num_layers*2, self.batch_size, self.hidden_dim // 2))

    def _get_lstm_features(self, sentences, lengths):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentences)
        embeds_packed = torch.nn.utils.rnn.pack_padded_sequence(embeds, 
                                                                lengths, 
                                                                batch_first=True)
        lstm_out_packed, self.hidden = self.lstm(embeds_packed, self.hidden)
        
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out_packed, batch_first=True)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def forward(self, sentences, lengths):
        # Get the emission scores from the BiLSTM
        lstm_feats_batch = self._get_lstm_features(sentences, lengths)

        return self.log_softmax(lstm_feats_batch)

    def neg_log_likelihood(self, sentences, lengths, tags_batch):
        
        output = self.forward(sentences, lengths).permute(0, 2, 1)
        loss_func = nn.NLLLoss()

        return loss_func(output, tags_batch)

    def predict(self, sentences, lengths):

        return torch.argmax(self.forward(sentences, lengths), dim=2)

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim,
                 lstm_num_layers=1, batch_size=2):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.lstm_num_layers = lstm_num_layers
        self.batch_size = batch_size

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=self.lstm_num_layers, bidirectional=True,
                            batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(self.lstm_num_layers*2, self.batch_size, self.hidden_dim // 2),
                torch.randn(self.lstm_num_layers*2, self.batch_size, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentences, lengths):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentences)
        embeds_packed = torch.nn.utils.rnn.pack_padded_sequence(embeds, 
                                                                lengths, 
                                                                batch_first=True)
        lstm_out_packed, self.hidden = self.lstm(embeds_packed, self.hidden)
        
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out_packed, batch_first=True)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], 
                                        dtype=torch.long), 
                          tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentences, lengths, tags_batch):
        feats_batch = self._get_lstm_features(sentences, lengths)

        score_diff = torch.full((feats_batch.shape[0], 1), 0.)
        for idx, feats in enumerate(feats_batch):
            unpadded_feats = feats[:lengths[idx]]
            unpadded_tags = tags_batch[idx][:lengths[idx]]
            forward_score = self._forward_alg(unpadded_feats)
            gold_score = self._score_sentence(unpadded_feats, unpadded_tags)
            score_diff[idx] = forward_score - gold_score
        return score_diff

    def forward(self, sentences, lengths):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats_batch = self._get_lstm_features(sentences, lengths)

        # Find the best path, given the features.
        score_batch = torch.full((lstm_feats_batch.shape[0], 1), 0.)
        tag_seq_batch = torch.full((lstm_feats_batch.shape[0], 
                                    lstm_feats_batch.shape[1]), 0, dtype=torch.long)
        for idx, lstm_feats in enumerate(lstm_feats_batch):
            score, tag_seq = self._viterbi_decode(lstm_feats)
            tag_seq = torch.tensor(tag_seq)
            score_batch[idx] = score
            tag_seq_batch[idx] = tag_seq
        return score_batch, tag_seq_batch


