from m2s.service.simple_server import create_app
from tagger.models.lstm import NerLSTM
from tagger.predictors.lstm import NerLSTMPredictor

model_path = '../pretrained/lstm/'
predictor = NerLSTMPredictor.load(NerLSTM, model_path)

# create service app
service_app = create_app(
            title='My LSTM NER Service',
            predictor=predictor)