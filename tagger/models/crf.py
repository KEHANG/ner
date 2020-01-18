import os
import json
import torch
import torch.nn as nn
from seqeval.metrics import f1_score
from torch.nn.utils.rnn import (pack_padded_sequence,
                                pad_packed_sequence)

import tagger.models.util as util
from tagger.models.base import NerBaseModel
from tagger.constant import START_TAG, STOP_TAG

torch.manual_seed(1)

class BiLSTM_CRF(NerBaseModel):

    def __init__(self, vocab_size,
                 tag_to_ix,
                 embedding_dim, hidden_dim,
                 lstm_num_layers=1):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.ix_to_tag = {self.tag_to_ix[tag] : tag for tag in self.tag_to_ix}
        self.tagset_size = len(tag_to_ix)
        self.lstm_num_layers = lstm_num_layers

        embedding_module = nn.Embedding(vocab_size, embedding_dim)

        encoder = nn.LSTM(embedding_dim,
                          hidden_dim // 2,
                          num_layers=self.lstm_num_layers,
                          bidirectional=True,
                          batch_first=True)

        # Maps the output of the LSTM into tag space.
        ner_heads = nn.Linear(hidden_dim, self.tagset_size)

        super(BiLSTM_CRF, self).__init__(embedding_module,
                                         encoder,
                                         ner_heads)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[:, tag_to_ix[START_TAG]] = -10000
        self.transitions.data[tag_to_ix[STOP_TAG], :] = -10000

    def rnn_forward(self, sentences, lengths):
        embeds = self.embedding_module(sentences)
        embeds_packed = pack_padded_sequence(embeds,
                                             lengths,
                                             batch_first=True)
        packed_activations, _ = self.encoder(embeds_packed)
        activations, _ = pad_packed_sequence(packed_activations,
                                             batch_first=True)
        outputs = self.ner_heads(activations)
        return outputs

    def _log_likelihood_numerator(self, logits, lengths, tags_batch):
        """
        calculate scores for tag sequences given logits. It's the logarithm of
        the numerator of P(s, x).

        Parameters
        ----------
        logits : torch.FloatTensor, required.
            shape = (batch_size, sequence_length, num_tags)
        lengths : torch.Tensor, required.
            shape = (batch_size, )
        tags_batch: torch.Tensor, required.
            shape = (batch_size, sequence_length)

        Outputs
        -------
        scores : torch.FloatTensor. shape = (batch_size, )
        """
        batch_size, sequence_length, _ = logits.data.shape

        # Transpose batch size and sequence dimensions:
        logits = logits.transpose(0, 1).contiguous()
        tags_batch = tags_batch.transpose(0, 1).contiguous()

        # Start with the transition scores from start_tag to the first tag in each input
        start_transitions = self.transitions.data[self.tag_to_ix[START_TAG], :]
        scores = start_transitions.index_select(0, tags_batch[0])

        # Add up the scores for the observed transitions and all the inputs but the last
        for i in range(sequence_length - 1):
            # Each is shape (batch_size,)
            current_tag, next_tag = tags_batch[i], tags_batch[i+1]

            # The scores for transitioning from current_tag to next_tag
            transition_score = self.transitions[current_tag.view(-1), next_tag.view(-1)]

            # The score for using current_tag
            emit_score = logits[i].gather(1, current_tag.view(batch_size, 1)).squeeze(1)

            # Include transition score if next element is unmasked,
            # input_score if this element is unmasked.
            this_mask = (lengths > i).to(dtype=torch.float32)
            next_mask = (lengths > (i + 1)).to(dtype=torch.float32)
            scores = scores + transition_score * next_mask + emit_score * this_mask

        # Transition from last state to "stop" state. To start with, we need to find the last tag
        # for each instance.
        last_tag_index = lengths - 1
        last_tags = tags_batch.gather(0, last_tag_index.view(1, batch_size)).squeeze(0)

        # Compute score of transitioning to `stop_tag` from each "last tag".
        stop_transitions = self.transitions.data[:, self.tag_to_ix[STOP_TAG]]
        last_transition_score = stop_transitions.index_select(0, last_tags)

        # Add the last input if it's not masked.
        last_inputs = logits[-1]                                         # (batch_size, num_tags)
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))  # (batch_size, 1)
        last_input_score = last_input_score.squeeze()                    # (batch_size,)

        last_mask = (lengths >= sequence_length).to(dtype=torch.float32)
        scores = scores + last_transition_score + last_input_score * last_mask

        return scores

    def _log_likelihood_denominator(self, logits, lengths):
        """
        calculate the logarithm of the denominator of P(s, x).

        Parameters
        ----------
        logits : torch.FloatTensor, required.
            shape = (batch_size, sequence_length, num_tags)
        lengths : torch.Tensor, required.
            shape = (batch_size, )

        Outputs
        -------
        scores : torch.FloatTensor. shape = (batch_size, )
        """
        batch_size, sequence_length, num_tags = logits.data.shape

        # Transpose batch size and sequence dimensions
        logits = logits.transpose(0, 1).contiguous()

        # Initial alpha is the (batch_size, num_tags) tensor of likelihoods combining the
        # transitions to the initial states and the logits for the first timestep.
        start_transitions = self.transitions.data[self.tag_to_ix[START_TAG], :]
        alpha = start_transitions.view(1, num_tags) + logits[0]

        # For each i we compute logits for the transitions from timestep i-1 to timestep i.
        # We do so in a (batch_size, num_tags, num_tags) tensor where the axes are
        # (instance, current_tag, next_tag)
        for i in range(1, sequence_length):
            # The emit scores are for time i ("next_tag") so we broadcast along the current_tag axis.
            emit_scores = logits[i].view(batch_size, 1, num_tags)
            # Transition scores are (current_tag, next_tag) so we broadcast along the instance axis.
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            # Alpha is for the current_tag, so we broadcast along the next_tag axis.
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)

            # Add all the scores together and logexp over the current_tag axis
            inner = broadcast_alpha + emit_scores + transition_scores

            # In valid positions (mask == 1) we want to take the logsumexp over the current_tag dimension
            # of ``inner``. Otherwise (mask == 0) we want to retain the previous alpha.
            this_mask = (lengths > i).to(dtype=torch.float32)
            alpha = (util.logsumexp(inner, 1) * this_mask.view(batch_size, 1) +
                     alpha * (1 - this_mask).view(batch_size, 1))

        # Every sequence needs to end with a transition to the stop_tag.
        stop_transitions = self.transitions.data[:, self.tag_to_ix[STOP_TAG]]
        stops = alpha + stop_transitions.view(1, num_tags)

        # Finally we log_sum_exp along the num_tags dim, result is (batch_size,)
        return util.logsumexp(stops)

    def forward(self, sentences, lengths, tags_batch=None):
        # run throught rnn layer of the model
        rnn_outputs = self.rnn_forward(sentences, lengths)

        if self.training:
            log_denominator = self._log_likelihood_denominator(rnn_outputs, lengths)
            log_numerator = self._log_likelihood_numerator(rnn_outputs, lengths, tags_batch)

            # return negative log likelihood
            return torch.sum(log_denominator - log_numerator)
        else:
            # extract important constants
            max_seq_length = rnn_outputs.shape[1]
            start_tag = self.tag_to_ix[START_TAG]
            end_tag = self.tag_to_ix[STOP_TAG]

            # Find the best path, given the rnn_outputs and transitions
            best_paths = []
            tag_sequence = torch.Tensor(max_seq_length + 2, self.tagset_size)
            for rnn_output, sequence_length in zip(rnn_outputs, lengths):
                # Start with everything totally unlikely
                tag_sequence.fill_(-10000.)
                # At timestep 0 we must have the START_TAG
                tag_sequence[0, start_tag] = 0.
                # At steps 1, ..., sequence_length we just use the incoming prediction
                tag_sequence[1:(sequence_length + 1), :self.tagset_size] = rnn_output[:sequence_length]
                # And at the last timestep we must have the END_TAG
                tag_sequence[sequence_length + 1, end_tag] = 0.

                viterbi_path, _ = util.viterbi_decode(
                                                tag_sequence[:(sequence_length + 2)],
                                                self.transitions
                                            )
                best_paths.append(torch.tensor(viterbi_path[1:sequence_length+1]))
            return best_paths

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

        # save model weights
        super(BiLSTM_CRF, self).save(output_dir)
        # save hyper-parameters
        model_params = {
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "lstm_num_layers": self.lstm_num_layers,
                "vocab_size": self.vocab_size,
                "tag_to_ix": self.tag_to_ix,
                "model_type": self.__class__.__name__
        }

        with open(os.path.join(output_dir, 'model_params.json'), 'w') as f_out:
            json.dump(model_params, f_out, indent=3)