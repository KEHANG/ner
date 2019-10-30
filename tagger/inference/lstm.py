
from tagger.models.lstm import NerLSTM
from tagger.predictors.lstm import NerLSTMPredictor

model_path = '../pretrained/lstm/'
predictor = NerLSTMPredictor.load(NerLSTM, model_path)

input_json = {'sentence':
                ['CRICKET','-','LEICESTERSHIRE','TAKE',
                 'OVER','AT','TOP','AFTER','INNINGS','VICTORY','.']}

tag_seq_pred= predictor.predict_json(input_json)
print(tag_seq_pred)