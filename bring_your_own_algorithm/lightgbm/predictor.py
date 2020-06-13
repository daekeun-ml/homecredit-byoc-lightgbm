import os
import json
import io
import flask
 
import numpy as np
import lightgbm as lgb
 
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model', 'lightgbm_model.txt')
 
class ScoringService(object):
    model = None
 
    @classmethod
    def get_model(cls):
        if cls.model == None:
            cls.model = lgb.Booster(model_file=model_path)
        return cls.model
 
    @classmethod
    def predict(cls, input):
        clf = cls.get_model()
        return clf.predict(input)
    
app = flask.Flask(__name__)
 
@app.route('/ping', methods=['GET'])
def ping():
    health = ScoringService.get_model() is not None
 
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    data = None
 
    if flask.request.content_type == 'text/csv':
        with io.StringIO(flask.request.data.decode('utf-8')) as f:
            data = np.loadtxt(f, delimiter=',')
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')
 
    print('Invoked with {} records'.format(data.shape[0]))
 
    predictions = ScoringService.predict(data)
 
    result = json.dumps({'results':predictions.tolist()})
 
    return flask.Response(response=result, status=200, mimetype='text/json')
