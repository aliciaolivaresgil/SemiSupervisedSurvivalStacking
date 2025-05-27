import scipy.stats as stats
import scikit_posthocs as sp
import cv2

import csv
import sys
import baycomp as bc

import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings 
import math

from utils.baycomp_plotting import tern 

names_map = {
    'logistic_regression': 'LR', 
    'trit_logistic_regression': 'TriT[LR]', 
    'random_forest': 'RF', 
    'trit_random_forest': 'TriT[RF]', 
    'ada_boost': 'AB', 
    'trit_ada_boost': 'TriT[AB]', 
    'knn': r'$k$-NN', 
    'trit_knn': r'TriT[$k$-NN]', 
    'svm': 'SVM', 
    'trit_svm': 'TriT[SVM]', 
}

def bayesian(model1, model2, data1, data2, metric, rope=0.05, type='narrow'): 

    data1 = np.array(data1)
    data2 = np.array(data2)
    posterior = bc.HierarchicalTest(data1, data2, rope=rope)
    with open(f'results_baycomp/bayesian_posteriors_rope={rope}_{model1}_{model2}_{metric}_{type}.pk', 'wb') as f: 
        pk.dump(posterior, f)

def generatePlots(model1, model2, metric, rope, type='narrow'): 
    posterior = pk.load(open(f'results_baycomp/bayesian_posteriors_rope={rope}_{model1}_{model2}_{metric}_{type}.pk', 'rb'))
    fig = tern(posterior, l_tag=names_map[model1], r_tag=names_map[model2])
    plt.savefig(f'figs/bayesian_rope={rope}_{model1}_{model2}_{metric}_{type}.pdf')

    matplotlib.pyplot.close()

if __name__=="__main__": 


    datasets_narrow = [
        'hdfail',  'Dialysis',  'dataOvarian1', 'dataDIVAT1', 'prostateSurvival', 'dataDIVAT3',  'nwtco', 'rott2', 'Aids2', 
        'LeukSurv', 'actg', 'UnempDur', 'scania', 'divorce',  'TRACE',  'nki70', 'micro.censure', 'phpl04K8a', 'zinc', 'whas500', 
        'dataDIVAT2', 'cost', 'rdata', 'GBSG2', 'grace', 'd.oropha.rec', 'retinopathy', 'ova', 'Unemployment', 'cgd', 'Z243', 
        'follic', 'burn', 'pharmacoSmoking', 'pbc', 'diabetes', 'veteran', 'Melanoma', 'e1684', 'Bergamaschi', 'breast', 
    ]
    
    datasets_wide = [
        'DLBCL', 
        'chop', 
        'DBCD', 
        'AML_Bull', 
        'vdv', 
        'NSBCD', 
        'MCLcleaned'
    ]

    model_pairs = [
        ('logistic_regression', 'trit_logistic_regression'),
        ('random_forest', 'trit_random_forest'), 
        ('ada_boost', 'trit_ada_boost'), 
        ('knn', 'trit_knn'),
        ('svm', 'trit_svm'), 
    ]

    metrics = ['cindex', 'brier']


    #NARROW DATASETS
    for metric in metrics: 
        for m1, m2 in model_pairs: 
            m1_data = []
            for dataset in datasets_narrow: 
                scores = pk.load(open(f'results/scores_dataset=({dataset})_model=({m1}).pk', 'rb'))
                if metric=='brier': 
                    m1_data.append([-s for s in scores[f'{metric}']])
                else: 
                    m1_data.append([s for s in scores[f'{metric}']])
            m2_data = []
            for dataset in datasets_narrow: 
                scores = pk.load(open(f'results/scores_dataset=({dataset})_model=({m2}).pk', 'rb'))
                if metric=='brier': 
                    m2_data.append([-s for s in scores[f'{metric}']])
                else: 
                    m2_data.append([s for s in scores[f'{metric}']])
            bayesian(m1, m2, m1_data, m2_data, metric, rope=0.05, type='narrow')
            generatePlots(m1, m2, metric, rope=0.05, type='narrow')


    #WIDE DATASETS
    for metric in metrics: 
        for m1, m2 in model_pairs: 
            m1_data = []
            for dataset in datasets_wide: 
                scores = pk.load(open(f'results/scores_dataset=({dataset})_model=({m1}).pk', 'rb'))
                if metric=='brier': 
                    m1_data.append([-s for s in scores[f'{metric}']])
                else: 
                    m1_data.append([s for s in scores[f'{metric}']])
            m2_data = []
            for dataset in datasets_wide: 
                scores = pk.load(open(f'results/scores_dataset=({dataset})_model=({m2}).pk', 'rb'))
                if metric=='brier': 
                    m2_data.append([-s for s in scores[f'{metric}']])
                else: 
                    m2_data.append([s for s in scores[f'{metric}']])
            bayesian(m1, m2, m1_data, m2_data, metric, rope=0.05, type='wide')
            generatePlots(m1, m2, metric, rope=0.05, type='wide')