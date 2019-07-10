import glob

import numpy as np
import matplotlib.pyplot as plt

import plot_utils

def plot_all_bert_stab_vs_dim():
	all_csv = glob.glob("./output/csv/stab_vs_dim*.csv")
	for csv_name in all_csv:
		groups, names, data = plot_utils.csv_to_table(csv_name)
		plt.figure()
		plot_utils.plot_figure_with_error_bar(names, data, color_list="b")
		# replace suffix and folder for pdf
		print(csv_name.replace(".csv", ".pdf").replace("csv", "figure"))
		plt.title(csv_name.split("dataset_")[1].split(".")[0], fontsize=17)
		plt.ylabel("Disagreement %", fontsize=17)
		plt.xlabel("Dimentionality", fontsize=17)
		plt.xscale("log")
		plt.savefig(csv_name.replace(".csv", ".pdf").replace("csv", "figure"))
		plt.savefig(csv_name.replace(".csv", ".png").replace("csv", "figure"))
		plt.close()

def plot_all_bert_stab_vs_comp():
	all_csv = glob.glob("./output/csv/stab_vs_comp*.csv")
	for csv_name in all_csv:
		groups, names, data = plot_utils.csv_to_table(csv_name)
		plt.figure()
		plot_utils.plot_figure_with_error_bar(names, data, color_list="b")
		# replace suffix and folder for pdf
		print(csv_name.replace(".csv", ".pdf").replace("csv", "figure"))
		plt.title(csv_name.split("dataset_")[1].split(".")[0], fontsize=17)
		plt.ylabel("Disagreement %", fontsize=17)
		plt.xlabel("precision", fontsize=17)
		plt.xscale("log")
		plt.savefig(csv_name.replace(".csv", ".pdf").replace("csv", "figure"))
		plt.savefig(csv_name.replace(".csv", ".png").replace("csv", "figure"))
		plt.close()

def plot_all_bert_stab_vs_ensemble():
	all_csv = glob.glob("./output/csv/stab_vs_ensemble*.csv")
	for csv_name in all_csv:
		groups, names, data = plot_utils.csv_to_table(csv_name)
		plt.figure()
		plot_utils.plot_figure_with_error_bar(names, data, color_list="b")
		# replace suffix and folder for pdf
		print(csv_name.replace(".csv", ".pdf").replace("csv", "figure"))
		plt.title(csv_name.split("dataset_")[1].split(".")[0], fontsize=17)
		plt.ylabel("Disagreement %", fontsize=17)
		plt.xlabel("Epsilon", fontsize=17)
		# plt.xscale("log")
		plt.savefig(csv_name.replace(".csv", ".pdf").replace("csv", "figure"))
		plt.savefig(csv_name.replace(".csv", ".png").replace("csv", "figure"))
		plt.close()

if __name__ == "__main__":
	# plot_all_bert_stab_vs_dim()
	# plot_all_bert_stab_vs_comp()
	plot_all_bert_stab_vs_ensemble()