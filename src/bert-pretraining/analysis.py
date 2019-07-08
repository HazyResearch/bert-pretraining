import sys, os
import glob
import json

import csv
import numpy as np
import utils
from plot_utils import save_csv_with_error_bar

def get_wiki17_wiki18_pred_disagreement_vs_dim(results, dims=[192, 384, 768, 1536, 3072], seeds=[1,2,3]):
    disagreements_all_dim = []
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
        disagreements_all_dim.append(disagreements)
        print("dim ", dim, "disagr. ave / std: ", np.mean(disagreements), np.std(disagreements))
    disagr = np.array(disagreements_all_dim).T
    data_list = [['Disagreement', dims, [disagr[i, :] for i in range(len(seeds))]]]
    return data_list

def print_all_stab_vs_dim_for_linear_bert_sentiment():
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
            data_list = get_wiki17_wiki18_pred_disagreement_vs_dim(results)
            print(data_list)
            csv_name = utils.get_csv_folder() + "/stab_vs_dim_{}_lr_dataset_{}.csv".format(exp_name, dataset)
            save_csv_with_error_bar(data_list, csv_name)


def get_wiki17_wiki18_pred_disagreement_generic(results, xlabel, xvalues, seeds=[1,2,3], subset_dict=None):
    disagrs_all= []
    for seed in seeds:
        disagrs = []
        for x in xvalues:
            # get wiki 17 results
            keys = {"corpus": ["wiki17"], "model_seed": [seed], xlabel: [x]}
            keys.update(subset_dict)

            #for test in results:
            #    print(test["corpus"], test["model_seed"], test["out"], xlabel)
            #    print(test["nbit"])
            subset = utils.extract_result_subset(results, keys)
            #print("keys ",keys, len(results), len(subset))
            print(subset[0]["test_err"])
            assert len(subset) == 1
            wiki17_pred = subset[0]["test_pred"]

            # get wiki 18 results
            keys = {"corpus": ["wiki18"], "model_seed": [seed], xlabel: [x]}
            keys.update(subset_dict)
            subset = utils.extract_result_subset(results, keys)
            print(subset[0]["test_err"])
            assert len(subset) == 1
            wiki18_pred = subset[0]["test_pred"]
            disagrs.append(utils.get_classification_disagreement(wiki17_pred, wiki18_pred))
        disagrs_all.append(disagrs)
        # print("dim ", dim, "disagr. ave / std: ", np.mean(disagreements), np.std(disagreements))
    disagr = np.array(disagrs_all)
    #print("all disagreement ", disagr.shape)
    data_list = [['Disagreement', xvalues, [disagr[i, :] for i in range(len(seeds))]]]
    return data_list


def print_all_stab_vs_compression_for_linear_bert_sentiment():
    datasets = ['mr', 'subj', 'mpqa', 'sst']
    exp_names = ['default', 'opt'] 
    for exp_name in exp_names:
        print("\n\n", exp_name)
        for dataset in datasets:    
            json_regex = "/home/zjian/bert-pretraining/results/predictions/compression_{}_lr_3_seeds_2019-07-08/{}/nbit_*/*/final_results.json".format(exp_name, dataset)
            # filter by dataset and setting
            results = utils.clean_json_results(utils.gather_json_results(json_regex))
            assert len(results) == 36, json_regex # 2 corpus x 3 seeds x 6 precision
            print("\n\n", dataset)
            data_list = get_wiki17_wiki18_pred_disagreement_generic(results, 
                xlabel="nbit", xvalues=[1,2,4,8,16,32], subset_dict={"feat_dim": [768]})
            print(data_list)
            csv_name = utils.get_csv_folder() + "/stab_vs_comp_{}_lr_dataset_{}.csv".format(exp_name, dataset)
            save_csv_with_error_bar(data_list, csv_name)


if __name__ == "__main__":
    # print_all_stab_vs_dim_for_linear_bert_sentiment()
    print_all_stab_vs_compression_for_linear_bert_sentiment()

    


