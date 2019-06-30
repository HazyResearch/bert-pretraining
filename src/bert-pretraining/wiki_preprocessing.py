import os, sys
import json
from glob import glob
import spacy
import re
import random
import math
from multiprocessing import Process

def get_raw_data_content(f_name):
	with open(f_name, "r") as f:
		content = f.readlines()
		content = [eval(x) for x in content]
	return content

def extract_article_id_name(path):
	files = glob(path + "/*/wiki*")
	id_to_name = {}
	id_to_len = {}
	for f_name in files:
		print("processing id and name from ", f_name)
		# with open(f_name, "r") as f:
		# 	content = f.readlines()
		# 	content = [eval(x) for x in content]
		content = get_raw_data_content(f_name)
		for x in content:
			x['id'] = int(x['id'])
			id_to_name[x['id']] = x['title']
			id_to_len[x['id']] = len(x['text'].split(' '))
	return id_to_name, id_to_len

def process_article_ids(path_wiki17, path_wiki18):
	id_to_name_wiki17, id_to_len_wiki17 = extract_article_id_name(path_wiki17)
	id_to_name_wiki18, id_to_len_wiki18 = extract_article_id_name(path_wiki18)
	with open("./output/id_to_name_wiki17", "w") as f:
		json.dump(id_to_name_wiki17, f)
	with open("./output/id_to_name_wiki18", "w") as f:
		json.dump(id_to_name_wiki18, f)
	with open("./output/id_to_len_wiki17", "w") as f:
		json.dump(id_to_len_wiki17, f)
	with open("./output/id_to_len_wiki18", "w") as f:
		json.dump(id_to_len_wiki18, f)

	print('start comparing ')
	token_cnt_wiki17 = 0
	token_cnt_wiki18 = 0
	# common_article_id = set(id_to_name_wiki17.keys()).intersection(id_to_name_wiki18.keys())
	# wiki18_only_article_id = set(id_to_name_wiki18.keys()) - set(id_to_name_wiki17.keys())
	for article_id in id_to_name_wiki17.keys():
		# if article_id in id_to_name_wiki18.keys():
		token_cnt_wiki17 += id_to_len_wiki17[article_id]
	for article_id in id_to_name_wiki18.keys():
		token_cnt_wiki18 += id_to_len_wiki18[article_id]
	print("# of tokens for 17 and 18 ", token_cnt_wiki17, token_cnt_wiki18)
	wiki17_id_list_sorted = sorted(list(id_to_name_wiki17.keys()))
	wiki18_id_list_sorted = sorted(list(id_to_name_wiki18.keys()))
	wiki17_id = set(wiki17_id_list_sorted)
	wiki18_id = set(wiki18_id_list_sorted)
	common_article_id = list(wiki17_id.intersection(wiki18_id))
	wiki18_only_article_id = list(wiki18_id.difference(wiki17_id))
	with open("./output/common_article_id", "w") as f:
		json.dump(common_article_id, f)
	with open("./output/wiki18_only_article_id", "w") as f:
		json.dump(wiki18_only_article_id, f)
	print("# of articles for 17 and 18, intersection / diff ", 
		len(id_to_name_wiki17.keys()), len(id_to_name_wiki18.keys()),
		len(common_article_id), len(wiki18_only_article_id))
	common_article_tokens = 0
	for article_id in common_article_id:
		common_article_tokens += id_to_len_wiki17[article_id]
	wiki18_only_article_tokens = 0
	for article_id in wiki18_only_article_id:
		wiki18_only_article_tokens += id_to_len_wiki18[article_id]
	print("17 18 # token intersection / diff ", common_article_tokens, wiki18_only_article_tokens)

def seg_json_sentences(spacy_proc, article, subset_ids):
	text = article['text']
	title = article['title']
	if int(article['id']) not in subset_ids:
		return []
	article = spacy_proc(text)
	sentences = []
	for i, sent in enumerate(article.sents):
		x = str(sent)
		# remove trailing white spaces
		x = x.rstrip(' ').rstrip('\n').rstrip('\r').rstrip('\t')
		if i == 0:
			# remove title from the begining of text
			x = x.replace(title + '\n\n', '')
		# remove in sentence \n
		x = x.replace('\n', '')
		sentences.append(x)
	return sentences

def proc_raw_data_file(f_in_name, f_out_name, subset_ids):
	spacy_proc = spacy.load("en")
	# given a wikiExtractor.py generated subfile, this func
	# produce the 1 sentence per line file
	with open(f_in_name, 'r') as f_in:
		with open(f_out_name, 'w') as f_out:
			content = get_raw_data_content(f_in_name)
			for x in content:
				# each iteration process an article
				sentences = seg_json_sentences(spacy_proc, x, subset_ids)
				for sent in sentences:
					f_out.write(sent + '\n')
				# separate articles
				if len(sentences) != 0:
					f_out.write('\n')
	# print("raw file processed ", f_in_name)

def subsample_wiki_id(article_id_dict="./output/common_article_id", subset_prop=0.1, seed=123):
	random.seed(seed)
	with open(article_id_dict, 'r') as f:
		article_ids = json.load(f)
	random.shuffle(article_ids)
	n_sample = math.floor(len(article_ids) * subset_prop)
	return article_ids[:n_sample]


if __name__ == "__main__":
	path_wiki17 = "/dfs/scratch0/zjian/bert-pretraining/data/wiki/wiki17/wiki_json"
	path_wiki18 = "/dfs/scratch0/zjian/bert-pretraining/data/wiki/wiki18/wiki_json"

	# # generate wiki dump article id related meta data
	# process_article_ids(path_wiki17, path_wiki18)

	# subsampling and get text file for tensorflow bert
	common_subset_ids = subsample_wiki_id("./output/common_article_id")
	wiki18_only_article_ids = subsample_wiki_id("./output/wiki18_only_article_id")
	wiki17_subset_ids = sorted(common_subset_ids)
	wiki18_subset_ids = sorted(common_subset_ids + wiki18_only_article_ids)

	for path, subset_ids in zip([path_wiki17, path_wiki18],
		[wiki17_subset_ids, wiki18_subset_ids]):
	# for path, subset_ids in zip([path_wiki17, ],
	# 	[wiki17_subset_ids, ]):
	# for path, subset_ids in zip([path_wiki18, ],
	# 	[wiki18_subset_ids, ]):
		raw_files = glob(path+'/*/wiki_*')
		# prevent old _sent files being processed
		raw_files = [x for x in raw_files if '_sent' not in x]
		proc_files = [x + "_sent" for x in raw_files]
		mprocs = []
		for i, (raw_file, proc_file) in enumerate(zip(raw_files, proc_files)):
			mproc = Process(target=proc_raw_data_file, 
				args=(raw_file, proc_file, subset_ids))
			mprocs.append(mproc)
			mproc.start()
			if i % 100 == 99 or i == len(raw_files) - 1:
				for mproc in mprocs:
					mproc.join()
				mprocs = []
				print("process done for ", path, " at ", i, " th sample out of ", len(raw_files))
	# # test example
	# f_in_name = "/dfs/scratch0/zjian/bert-pretraining/data/wiki/wiki17/wiki_json/AA/wiki_01"
	# f_out_name = "/dfs/scratch0/zjian/bert-pretraining/data/wiki/wiki17/wiki_json/AA/sent_wiki_01"
	# proc_raw_data_file(f_in_name, f_out_name, wiki17_subset_ids)

