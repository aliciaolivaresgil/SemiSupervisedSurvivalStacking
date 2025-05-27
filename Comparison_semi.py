import pandas as pd
import numpy as np
from random import random
import pickle as pk
from SurvSet.data import SurvLoader

from sklearn import preprocessing
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sslearn.wrapper import TriTraining

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV

from sksurv.metrics import concordance_index_censored
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.metrics import integrated_brier_score

from utils.survival_stacking import stack_timepoints_semi, stack_eval, cumulative_hazard_function, risk_score, filter_times_on_test, survival_function

def repeatedCrossVal(estimator, param_grid, X, y, n_splits, random_state): 
    
    results_dict = dict()
    results_dict['cindex'] = []
    results_dict['times'] = []
    results_dict['auc'] = []
    results_dict['mean_auc'] = []
    results_dict['brier'] = []
    tuned_params = []
    predictions = []
    
    outer_cv = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    
    #calculate bins 
    n_bins = 10
    t_max = max([t for e,t in y])
    event_times = np.linspace(0, t_max, n_bins+1)[1:]
    
    time_bin = []
    for e,t in y: 
        time_bin.append(f'{min([et for et in event_times if t<=et])}_{e}')

    for i, (train_index, test_index) in enumerate(outer_cv.split(X, time_bin)):
        print(f'\tSplit {i}')

        X_train = X.iloc[train_index].to_numpy()
        X_test = X.iloc[test_index].to_numpy()
        y_train = y[train_index]
        y_test = y[test_index]

        X_fun_train, y_fun_train = stack_timepoints_semi(X_train, y_train, event_times)
        X_fun_test = stack_eval(X_test, event_times)

        inner_cv = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
        grid_search = GridSearchCV(model, param_grid=param_grid, cv=inner_cv, verbose=1)
        result = grid_search.fit(X_fun_train, y_fun_train)
        best_model = result.best_estimator_

        f_test_estimates = result.predict_proba(X_fun_test)[:,1]
        f_test_chf = cumulative_hazard_function(f_test_estimates, event_times)
        f_test_risk = risk_score(f_test_estimates, event_times)
        
        #c-index
        events_test = [e for e,t in y_test]
        times_test = [t for e,t in y_test]
        cindex, _, _, _, _ = concordance_index_censored(events_test, times_test, f_test_risk)
        results_dict['cindex'].append(cindex)
    
        #cumulative dynamic auc 
        f_filt_times, f_filt_chf = filter_times_on_test(y_test, f_test_chf, event_times)
        y_train_test = np.concatenate((y_train, y_test), axis=0)
        _auc, _mean_auc = cumulative_dynamic_auc(survival_train=y_train_test, survival_test=y_test, 
                                                 estimate=f_filt_chf, times=f_filt_times)
        results_dict['auc'].append(_auc)
        results_dict['mean_auc'].append(_mean_auc)
        results_dict['times'].append(f_filt_times)

        #integrated brier score
        survs = survival_function(f_test_estimates, event_times)
        min_t = min([t for _,t in y_test])
        max_t = max([t for _,t in y_test])
        min_surv = min([t for t in survs[0].x])
        max_surv = max([t for t in survs[0].x])
        min_t = max([min_t, min_surv])
        max_t = min([max_t, max_surv])
        
        _event_times = [et for et in event_times if et<max_t and et>=min_t]
        print(_event_times)
        preds = np.asarray([[fn(t) for t in _event_times] for fn in survs])
        brier_score = integrated_brier_score(y_train_test, y_test, preds, _event_times)
        results_dict['brier'].append(brier_score)
        
    return results_dict, predictions, tuned_params

if __name__=="__main__": 
    
    datasets = [
        'hdfail',  'Dialysis',  'dataOvarian1', 'dataDIVAT1', 'prostateSurvival', 'dataDIVAT3',  'nwtco', 'rott2', 'Aids2', 
        'LeukSurv', 'actg', 'UnempDur', 'scania', 'divorce',  'TRACE',  'nki70', 'micro.censure', 'phpl04K8a', 'zinc', 'whas500', 
        'dataDIVAT2', 'cost', 'rdata', 'GBSG2', 'grace', 'd.oropha.rec', 'retinopathy', 'ova', 'Unemployment', 'cgd', 'Z243', 
        'follic', 'burn', 'pharmacoSmoking', 'pbc', 'diabetes', 'veteran', 'Melanoma', 'e1684', 'Bergamaschi', 'breast', 
    ]
    datasets = [
        'DLBCL', 
        'chop', 
        'DBCD', 
        'AML_Bull', 
        'vdv', 
        'NSBCD', 
        'MCLcleaned'
    ] 
  
    random_state = 4444

    for dataset in datasets: 
        loader = SurvLoader()
        df, ref = loader.load_dataset(ds_name=dataset).values()

        #PREPROCESS
        categorical = []
        numeric = []
        y = df[['event', 'time']]
        y = np.array([(i, t) for i,t in y[['event', 'time']].to_numpy()], dtype=[('Ingreso', '?'), ('t', '<f8')])
        X = df.drop(['pid', 'event', 'time'], axis=1)
        for column in X:
            if X[column].dtype == 'object': 
                categorical.append(column)
            else: 
                numeric.append(column)
        #min-max normalization
        X_numeric = (X[numeric]-X[numeric].min()) / (X[numeric].max()-X[numeric].min())
        #ohe
        if categorical: 
            X_categorical = pd.get_dummies(X[categorical])
            X = pd.concat([X_numeric, X_categorical], axis=1)
        else: 
            X = X_numeric

        models = {
            'trit_logistic_regression': (TriTraining(base_estimator=LogisticRegression(max_iter=5000)),
                     {"base_estimator__C": np.logspace(-3,3,7)}),
            'trit_random_forest': (TriTraining(base_estimator=RandomForestClassifier()), {}),
            'trit_ada_boost': (TriTraining(base_estimator=AdaBoostClassifier()), {}), 
            'trit_svm': (TriTraining(base_estimator=SVC(probability=True)), {}), 
            'trit_knn': (TriTraining(base_estimator=KNeighborsClassifier()), {})
        }

        for key_model in models: 
            model, param_grid = models[key_model]
            print(f'DATASET-> {dataset}, MODEL-> {key_model}')

            results_dict, predictions, tuned_params = repeatedCrossVal(model, param_grid, X, y, 
                                                                   n_splits=10, random_state=random_state)
            
            with open(f'results/scores_dataset=({dataset})_model=({key_model}).pk', 'wb') as f: 
                pk.dump(results_dict, f)
            with open(f'results/predictions_dataset=({dataset})_model=({key_model}).pk', 'wb') as f: 
                pk.dump(predictions, f)
            with open(f'results/tuned_params_dataset=({dataset})_model=({key_model}).pk', 'wb') as f: 
                pk.dump(tuned_params, f)

