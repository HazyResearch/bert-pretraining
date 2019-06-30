import os, sys
import json
from glob import glob

def extract_article_id_name(path):
	files = glob(path + "/*/wiki*")
	id_to_name = {}
	id_to_len = {}
	for f_name in files:
		print("processing id and name from ", f_name)
		with open(f_name, "r") as f:
			content = f.readlines()
			content = [eval(x) for x in content]
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


def seg_sentences(spacy_proc, text):
	article = spacy_proc(text, parse=True)
	return article.sents



if __name__ == "__main__":
	path_wiki17 = "/dfs/scratch0/zjian/bert-pretraining/data/wiki/wiki17/wiki_json"
	path_wiki18 = "/dfs/scratch0/zjian/bert-pretraining/data/wiki/wiki18/wiki_json"
	process_article_ids(path_wiki17, path_wiki18)

