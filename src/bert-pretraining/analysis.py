import sys, os
import glob
import json

import numpy as np
import utils

def print_wiki17_wiki18_pred_disagreement_vs_dim(results, dims=[192, 384, 768, 1536, 3072], seeds=[1,2,3]):
    for dim in dims:
        disagreements = []
        for seed in seeds:
            # get wiki 17 results
            keys = {"corpus": ["wiki17"], "feat_dim": [dim], "model_seed": [seed]}
            subset = utils.extract_result_subset(results, keys)
            print(subset[0]["test_err"])
            assert len(subset) == 1
            wiki17_pred = subset[0]["test_pred"]

            # get wiki 18 results
            keys = {"corpus": ["wiki18"], "feat_dim": [dim], "model_seed": [seed]}
            subset = utils.extract_result_subset(results, keys)
            print(subset[0]["test_err"])
            assert len(subset) == 1
            wiki18_pred = subset[0]["test_pred"]
            disagreements.append(utils.get_classification_disagreement(wiki17_pred, wiki18_pred))
        print("dim ", dim, "disagr. ave / std: ", np.mean(disagreements), np.std(disagreements))

def generate_all_predictions_for_linear_bert_sentiment():
    datasets = ['mr', 'subj', 'mpqa', 'sst']
    nbit = 32   
    exp_names = ['default', 'opt'] 
    for exp_name in exp_names:
        print("\n\n", exp_name)
        for dataset in datasets:    
            json_regex = "/home/zjian/bert-pretraining/results/predictions/dimensionality_{}_lr_3_seeds_2019-07-07/{}/nbit_32/*/final_results.json".format(exp_name, dataset)
            # filter by dataset and setting
            results = utils.clean_json_results(utils.gather_json_results(json_regex))
            assert len(results) == 30, json_regex
            print("\n\n", dataset)
            print_wiki17_wiki18_pred_disagreement_vs_dim(results)


if __name__ == "__main__":
    generate_all_predictions_for_linear_bert_sentiment()

    


