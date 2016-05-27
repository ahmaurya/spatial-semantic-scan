import fileinput
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statistics
from matplotlib.backends.backend_pdf import PdfPages

log_file = '../results/simulation_tss/intensity_vs_metrics.log'
x_variable = 'Emerging Topic Intensity'
x_axis = 'Emerging Topic Intensity'

#log_file = '../results/simulation_tss/k_vs_metrics.log'
#x_variable = 'K (Neighborhood Size in GFSS)'
#x_axis = 'K (Neighborhood Size in GFSS in Hundreds)'

plot_titles = ['X', 'I', 'KL_0', 'KL_1', 'TN_KL_00', 'TN_KL_01', 'TN_KL_10', 'TN_KL_11',
		'$l_1$ Difference', 'Truncated $l_1$ Difference', 
		'$l_\infty$ Difference', 'Truncated $l_\infty$ Difference', 
		'Spatial Overlap', 'Precision', 'Recall', 'Extra 1', 'Extra 2', 'Extra 3']

plot_xlabels = [x_axis for i in range(len(plot_titles))]

plot_ylabels = ['X', 'I',  'KL Divergence w.r.t. True Topic', 'KL Divergence w.r.t. Estimated Topic',
		'Truncated KL Divergence 00', 'Truncated KL Divergence 01', 'Truncated KL Divergence 10', 'Truncated KL Divergence 11', 
		'$l_1$ difference between true and estimated topics', 'Truncated $l_1$ difference between true and estimated topics', 
		'$l_\infty$ difference between true and estimated topics', 'Truncated $l_\infty$ difference between true and estimated topics', 
		'Spatial Overlap', 'Precision', 'Recall', 'Extra 1', 'Extra 2', 'Extra 3']

lines = []
for line in fileinput.input(log_file):
	line = line.strip().split()
	line = [float(x) for x in line]
	lines.append(line)

# metrics versus x variable
for field in range(len(line)):
	xs = sorted(list(set([line[1] for line in lines])))
	ys = [[line[field] for line in lines if line[1]==x] for x in xs]
	means = [statistics.mean(ys[i]) for i in range(len(xs))]
	errors = [statistics.variance(ys[i]) for i in range(len(xs))]

	plt.figure()
	plt.errorbar(xs, means, yerr=errors)
	plt.title(plot_titles[field] + ' versus ' + x_variable)
	plt.xlabel(plot_xlabels[field])
	plt.ylabel(plot_ylabels[field])
	
	#plt.xlim([0,2500])
	#plt.ylim([0.0,1.1])
	#plt.xticks(np.arange(0,2600,100))
	#plt.yticks(np.arange(0,1.2,0.1))
	plt.grid()
	
	plt.show()
	#pp = PdfPages(str(field)+'.pdf')
	#pp.savefig()
	#pp.close()

# precision versus recall
xs = sorted(list(set([line[1] for line in lines])))
precisions = [[line[13] for line in lines if line[1]==x] for x in xs]
precision_means = [statistics.mean(precisions[i]) for i in range(len(xs))]
precision_errors = [statistics.variance(precisions[i]) for i in range(len(xs))]
recalls = [[line[14] for line in lines if line[1]==x] for x in xs]
recall_means = [statistics.mean(recalls[i]) for i in range(len(xs))]
recall_errors = [statistics.variance(recalls[i]) for i in range(len(xs))]

plt.figure()
plt.errorbar(precision_means, recall_means, xerr=precision_errors, yerr=recall_errors)
plt.title('Precision versus Recall (Varying Emerging Topic Intensity)')
plt.xlabel('Precision')
plt.ylabel('Recall')

plt.xlim([0.0,1.1])
plt.ylim([0.0,1.1])
plt.xticks(np.arange(0,1.2,0.1))
plt.yticks(np.arange(0,1.2,0.1))
plt.grid()

plt.show()
#pp = PdfPages('prec_vs_recall.pdf')
#pp.savefig()
#pp.close()


