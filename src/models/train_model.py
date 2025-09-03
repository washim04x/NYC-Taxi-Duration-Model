from pathlib import Path
import joblib
import yaml
import pandas as pd
import sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.model_selection import train_test_split
from dvclive import Live


def train_model(train_features,target,params,live):
    Xtr, Xv, ytr, yv = train_test_split(train_features.values,target, test_size=params['test_size'], random_state=params['random_state'])
    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dvalid = xgb.DMatrix(Xv, label=yv)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

    # Try different parameters! My favorite is random search :)
    xgb_pars = {'min_child_weight': params['min_child_weight'], 'eta': params['eta'], 'colsample_bytree': params['colsample_bytree'], 'max_depth': params['max_depth'],
                'subsample': params['subsample'], 'lambda': params['lambda'], 'nthread': params['nthread'], 'booster' : params['booster'], 'silent': params['silent'],
                'eval_metric': params['eval_metric'], 'objective': params['objective']}
    model = xgb.train(xgb_pars, dtrain, params['num_boost_round'], watchlist, early_stopping_rounds=params['early_stopping_rounds'],
                  maximize=False,verbose_eval=params['verbose_eval'])
    # Log best_iteration and best_score as metrics for DVC
    live.log_metric("best_iteration", model.best_iteration)
    live.log_metric("best_score", model.best_score)



    return model


def save_model(model, output_path):
    # Save the trained model to the specified output path
    joblib.dump(model, output_path + '/model.joblib')

def main():
    curr_dir=Path(__file__)
    home_dir=curr_dir.parent.parent.parent
    params_path=home_dir.as_posix()+'/params.yaml'
    params=yaml.safe_load(open(params_path))["train_model"]

    input_file=sys.argv[1] #/data/processed
    data_path=home_dir.as_posix()+input_file
    output_path=home_dir.as_posix()+'/models'
    Path(output_path).mkdir(parents=True, exist_ok=True)

    
    TARGET = 'trip_duration'
    train_features = pd.read_csv(data_path + '/train.csv')
    X = train_features.drop(TARGET, axis=1)
    y = np.log(train_features[TARGET].values+1)
   
    with Live("dvclive",dvcyaml=False ) as live:
        trained_model = train_model(X, y, params,live)

    save_model(trained_model, output_path)




    

if __name__ == "__main__":
    main()
