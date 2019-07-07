import glob

import numpy as np
import matplotlib.pyplot as plt

import plot_utils

def plot_all_bert_stab_vs_dim():
	all_csv = glob.glob("./output/*stab_vs_dim*.csv")
	for csv_name in all_csv:
		groups, names, data = plot_utils.csv_to_table(csv_name)
		plt.figure()
		plot_figure_with_error_bar(names, data, color_list="b")
		# replace suffix and folder for pdf
		plt.save_fig(csv_name.replace(".csv", ".pdf").replace("csv", "figure"))
		plt.close()

if __name__ == "__main__":
	plot_all_bert_stab_vs_dim()