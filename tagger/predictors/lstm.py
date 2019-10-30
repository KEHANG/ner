import os
import json
import torch
from overrides import overrides
from m2s.predictors.predictor import Predictor

class NerLSTMPredictor(Predictor):
    """
    a ``NerLSTMPredictor`` predicts NER sequence
    using a LSTM model.
    """
    @overrides
    def predict_json(self, input_json):
        instance = self._json_to_instance(input_json)
        tag_seq_pred = self.predict_instance(instance)
        readable_tag_seq = [self.ix_to_tag[ix] for ix in tag_seq_pred[0]]
        return readable_tag_seq

    @overrides
    def _json_to_instance(self, input_json):
        """
        Expects JSON that looks like
        ``{"x": list[float]}``.
        """
        if "sentence" not in input_json:
            raise AssertionError("Expect sentence to be in the input, but not there.")
        input_sentence = input_json["sentence"]
        sentence = torch.tensor([self.word_to_ix[word] for word in input_sentence])
        length = torch.tensor([sentence.shape[0]])
        sentence = torch.unsqueeze(sentence, 0)
        print(sentence, length)
        instance = (sentence, length)
        return instance

    @classmethod
    def load(predictor_cls, model_cls, model_path):
        model = model_cls.load(model_path)
        predictor = predictor_cls(model)
        with open(os.path.join(model_path, 'word_to_ix.json'), 'r') as f:
            predictor.word_to_ix = json.load(f)

        with open(os.path.join(model_path, 'tag_to_ix.json'), 'r') as f:
            tag_to_ix = json.load(f)

        predictor.ix_to_tag = {tag_to_ix[tag] : tag for tag in tag_to_ix}

        return predictor