import numpy as np
import torch
import argparse
import sys, os
import logging

from smallfry import compress

import utils
NUM_TOL = 1e-6

def read_npy_feature(file):
    feats = np.load(file)
    feats = [feats[name] for name in feats.files]
    return feats

def read_npy_label(file):
    label = np.load(file)
    return label

def save_final_results(args, range_limit):
    results = args.__dict__
    results["results"] = {"range_limit": range_limit}
    utils.save_to_json(results, args.out_folder + "/final_results.json")

def main():
    # add arguments
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--input_file", type=str, help="The feature file to be compressed.")
    argparser.add_argument("--out_folder", type=str, help="The folder to contain the output")
    argparser.add_argument("--nbit", type=int, help="Number of bits for compressed features.")
    argparser.add_argument("--dataset", type=str, help="The dataset for asserting on the filenames. Only for on the fly check")
    argparser.add_argument("--seed", type=int, help="Random seeds for the sampleing process.")
    argparser.add_argument("--golden_sec_tol", type=float, default=1e-3,
        help="termination criterion for golden section search")
    args = argparser.parse_args()

    # assert args.dataset in args.input_file
    # assert "seed_{}".format(args.seed) in args.input_file

    utils.ensure_dir(args.out_folder)
    utils.init_logging(args.out_folder)

    # set random seeds
    utils.set_random_seed(args.seed)

    # load the dataset
    feats = read_npy_feature(args.input_file)
    labels = read_npy_label(args.input_file.replace(".feature.npz", ".label.npy"))

    # we will directly save if we need 32 bit embeddings
    range_limit = None
    if args.nbit != 32:
        # subsample and concatenate
        subset_id = np.random.choice(np.arange(len(feats)), size=int(len(feats) * 0.1))
        feats_subset = [feats[i].copy() for i in subset_id]
        X = np.concatenate(feats_subset, axis=0)

        # assert the last axis is feature dimension
        assert feats_subset[0].shape[-1] == feats_subset[-1].shape[-1]
        assert feats_subset[1].shape[-1] == feats_subset[-1].shape[-1]
        
        # run the compression to each of the things in the list
        logging.info("Estimate range limit using sample shape " + str(X.shape))
        range_limit = compress.find_optimal_range(X, args.nbit, stochastic_round=False, tol=args.golden_sec_tol)
        logging.info("Range limit {}, max/min {}/{}, std {} ".format(
            range_limit, np.max(X), np.min(X), np.std(X)))
        compressed_feats = []
        for i, feat in enumerate(feats):
            comp_feat = compress._compress_uniform(feat, args.nbit, range_limit,
                stochastic_round=False, skip_quantize=False)
            np.copyto(dst=feat, src=comp_feat)
            assert np.max(feats[i]) - range_limit < NUM_TOL, "not clipped right max/limit {}/{}".format(np.max(feats[i]), range_limit)
            assert -range_limit - np.min(feats[i]) < NUM_TOL, "not clipped right max/limit {}/{}".format(np.min(feats[i]), -range_limit)
            assert np.unique(feats[i]).size <= 2**args.nbit, "more unique values than expected"
    # save the results back to the format
    out_file_name = os.path.basename(args.input_file)
    out_file_name = args.out_folder + "/" + out_file_name
    np.savez(out_file_name, *feats)
    np.save(out_file_name.replace(".feature.npz", ".label.npy"), labels)
    save_final_results(args, range_limit)

    # TODO: the range limit is correct, compression is indeed carried out and it is properly copyed inplace
    # TODO: check a direct through saving case, check a compressed case to see the similarity
    # sanity check on if the tol is reasoanble

if __name__ == "__main__":
    main()
    
