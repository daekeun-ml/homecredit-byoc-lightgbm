#!/usr/bin/env python3

from __future__ import print_function

import argparse
import logging
import os
import json
import pickle
import sys
import traceback
import lightgbm as lgb

# These are the paths to where SageMaker mounts interesting things in your container.
prefix = '/opt/ml/'
# input_path = prefix + 'input/data'
# output_path = os.path.join(prefix, 'output')
# model_path = os.path.join(prefix, 'model')
param_dir = os.path.join(prefix, 'input/config/hyperparameters.json')
# inputdataconfig_dir = os.path.join(prefix, 'input/config/inputdataconfig.json')

# "val": "/opt/ml/input/data/val",
# "test": "/opt/ml/input/data/test",
# "calib": "/opt/ml/input/data/calib",
# "train": "/opt/ml/input/data/train"
            
channel_names = ['train', 'valid']

# The function to execute the training.
def train(train_dir, valid_dir, model_dir, output_dir):
    print('Starting the training.')
    try:
        # Read in any hyperparameters that the user passed with the training job
        # Read in any hyperparameters that the user passed with the training job
        #with open(param_dir, 'r') as f:
        #    hyperparams = json.load(f)
        hyperparams = {'num_round': 10,
            'objective':'multiclass',
            'num_class':3
        }
        

        train_filepath = os.path.join(train_dir, 'train.bin')
        valid_filepath = os.path.join(valid_dir, 'valid.bin')
        print(train_filepath, valid_filepath)
        dtrain = lgb.Dataset(train_filepath)
        dvalid = lgb.Dataset(valid_filepath)       

        valid_list = [dtrain, dvalid]
        
#         if 'valid' in inputdata_dic:
#             dvalid = inputdata_dic['valid']
#             valid_list = [dtrain, dvalid]
#         else:
#             valid_list = [dtrain]
            
        print('training start')    
        # Training
        model = lgb.train(
            params=hyperparams, 
            train_set=dtrain,
            valid_sets=valid_list
        )
        
        print('training complete')
        # Save the model
        model.save_model(os.path.join(model_dir, 'lightgbm_model.txt'))
     
        # The trained model can also be dumped to JSON format
        #json_model = model.dump_model()
        
        print('Training complete.')

    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_dir, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)





if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--valid_dir', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))    

    
    args, _ = parser.parse_known_args()

    train(args.train_dir, args.valid_dir, args.model_dir, args.output_dir)

    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)


# %%writefile catboost_training.py

# import argparse
# import logging
# import os

# from catboost import CatBoostRegressor
# import numpy as np
# import pandas as pd


# if __name__ =='__main__':

#     print('extracting arguments')
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
#     parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
#     parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
#     parser.add_argument('--train-file', type=str, default='boston_train.csv')
#     parser.add_argument('--test-file', type=str, default='boston_test.csv')
#     parser.add_argument('--model-name', type=str, default='catboost_model.dump')
#     parser.add_argument('--features', type=str)  # in this script we ask user to explicitly name features
#     parser.add_argument('--target', type=str) # in this script we ask user to explicitly name the target

#     args, _ = parser.parse_known_args()

#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)
    
#     logging.info('reading data')
#     train_df = pd.read_csv(os.path.join(args.train, args.train_file))
#     test_df = pd.read_csv(os.path.join(args.test, args.test_file))

#     logging.info('building training and testing datasets')
#     X_train = train_df[args.features.split()]
#     X_test = test_df[args.features.split()]
#     y_train = train_df[args.target]
#     y_test = test_df[args.target]
        
#     # define and train model
#     model = CatBoostRegressor()
    
#     model.fit(X_train, y_train, eval_set=(X_test, y_test), logging_level='Silent') 
    
#     # print abs error
#     logging.info('validating model')
#     abs_err = np.abs(model.predict(X_test) - y_test)

#     # print couple perf metrics
#     for q in [10, 50, 90]:
#         logging.info('AE-at-' + str(q) + 'th-percentile: '
#               + str(np.percentile(a=abs_err, q=q)))
    
#     # persist model
#     path = os.path.join(args.model_dir, args.model_name)
#     logging.info('saving to {}'.format(path))
#     model.save_model(path)