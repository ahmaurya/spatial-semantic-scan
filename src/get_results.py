import fileinput
from optparse import OptionParser, Option
from statistics import *

parser = OptionParser()
parser.add_option("-d", "--dir", dest="dir", type="string", default='../../semantic_scan/results/edc_ss/', help="Directory")
parser.add_option("-i", "--inputfile", dest="inputfile", type="string", default="metrics-labeled-lda-{icd9code}-{inject}.csv", help="Input File")
parser.add_option("-j", "--outputfile", dest="outputfile", type="string", default="metrics-labeled-lda.csv", help="Output File")
parser.add_option("-o", "--outputbydayfile", dest="outputbydayfile", type="string", default="metrics-by-day-labeled-lda.csv", help="Output by Day File")
parser.add_option("-f", "--outputbyfprfile", dest="outputbyfprfile", type="string", default="metrics-by-fpr-labeled-lda.csv", help="Output by False Positive Rate File")
parser.add_option("-m", "--maxinjectdays", dest="maxinjectdays", type="int", default=10, help="Maximum number of days in an inject")
parser.add_option("-b", "--maxbaselinedays", dest="maxbaselinedays", type="int", default=200, help="Maximum number of baseline days")
parser.add_option("-n", "--numinjects", dest="numinjects", type="int", default=10, help="Number of different injects to be generated per ICD9Code")
parser.add_option("-w", "--windowdays", dest="windowdays", type="int", default=3, help="Window days")
(options, args) = parser.parse_args()

input_path = options.dir + options.inputfile
results_path = options.dir + options.outputfile
results_by_day_path = options.dir + options.outputbydayfile
results_by_fpr_path = options.dir + options.outputbyfprfile

#icd9codes = ['mexican', 'chinese', 'german', 'argentine', 'vegan', 'szechuan', 'moroccan', 'malaysian', 'lebanese', 'irish', 'thai', 'russian', 'italian', 'portuguese', 'taiwanese', 'japanese', 'caribbean', 'indian', 'ukrainian', 'colombian', 'cantonese', 'french', 'filipino', 'vietnamese', 'pakistani', 'british', 'kosher', 'ethiopian', 'spanish', 'brazilian', 'mediterranean', 'peruvian', 'salvadoran', 'greek', 'korean', 'shanghainese', 'southern', 'pubs', 'afghan', 'cuban', 'venezuelan', 'seafood', 'african']
icd9codes = ['786', '780', '789', '729', '959', '724', '719', '787', '784', 'v72']
metrics_by_day = [{'spatial_precision': [], 'spatial_recall': [], 'spatial_overlap': [], 
	'topic_precision': [], 'topic_recall': [], 'topic_overlap': [], 'hellinger': []} for day in range(options.maxinjectdays)]
metrics_by_fpr = [{'num_detected': [], 'days_to_detect': []} for fpr in range(options.maxbaselinedays)]
metrics_by_day_mean = [{'spatial_precision': 0.0, 'spatial_recall': 0.0, 'spatial_overlap': 0.0, 
	'topic_precision': 0.0, 'topic_recall': 0.0, 'topic_overlap': 0.0, 'hellinger': 0.0} for day in range(options.maxinjectdays)]
metrics_by_fpr_mean = [{'num_detected': 0.0, 'days_to_detect': 0.0} for fpr in range(options.maxbaselinedays)]
metrics_by_day_stdev = [{'spatial_precision': 0.0, 'spatial_recall': 0.0, 'spatial_overlap': 0.0, 
	'topic_precision': 0.0, 'topic_recall': 0.0, 'topic_overlap': 0.0, 'hellinger': 0.0} for day in range(options.maxinjectdays)]
metrics_by_fpr_stdev = [{'num_detected': 0.0, 'days_to_detect': 0.0} for fpr in range(options.maxbaselinedays)]
runtimes = []
runtimes_mean = 0.0
runtimes_stdev = 0.0

for icd9code in icd9codes:
	for inject in range(options.numinjects):
		# get records from file
		records = []
		runtime = 0.0
		for line in fileinput.input(input_path.format(icd9code=icd9code, inject=inject)):
			line = line.strip().split(',')
			try:
				records.append({
					'spatial_precision': float(line[0]), 'spatial_recall': float(line[1]), 'spatial_overlap': float(line[2]), 
					'topic_precision': float(line[6]), 'topic_recall': float(line[7]), 'topic_overlap': float(line[8]), 
					'heldout_icd9code': str(line[9]), 'start_day': int(line[10]), 'inject_start_day': int(line[11]), 'inject_end_day': int(line[12]),
					'window_days': int(line[13]), 'baseline_days': int(line[14]), 'score': float(line[15]),
					'hellinger': float(line[16]), 'runtime': abs(float(line[17]))
					})
			except Exception as e:
				pass

		if len(records) < options.maxbaselinedays + options.maxinjectdays:
			continue

		# calculate stuff for getting further results
		sorted_baseline_scores = list(sorted([record['score'] for record in records if record['start_day'] < record['inject_start_day'] - 2 or record['start_day'] > record['inject_end_day'] + 2], reverse=True))

		# add to metrics_by_day
		for record in records:
			if record['start_day'] >= record['inject_start_day'] - options.windowdays + 1 and record['start_day'] <= record['inject_end_day'] - options.windowdays + 1:
				start_day = record['start_day'] - record['inject_start_day'] + options.windowdays - 1
				for key in ['spatial_precision', 'spatial_recall', 'spatial_overlap', 'topic_precision', 'topic_recall', 'topic_overlap', 'hellinger']:
					metrics_by_day[start_day][key].append(record[key])
			runtime += record['runtime']

		for fpr in range(options.maxbaselinedays):
			max_baseline_score = sorted_baseline_scores[fpr+1]
			detected = False
			for record in records:
				if record['start_day'] >= record['inject_start_day'] - options.windowdays + 1 and record['start_day'] <= record['inject_end_day'] and detected == False:
					start_day = record['start_day'] - record['inject_start_day'] + options.windowdays
					score = record['score']
					if detected == False and score > max_baseline_score:
						detected = True
						days_to_detect = start_day
						break

			if detected:
				metrics_by_fpr[fpr]['days_to_detect'].append(days_to_detect)
				metrics_by_fpr[fpr]['num_detected'].append(1)
			else:
				metrics_by_fpr[fpr]['days_to_detect'].append(options.maxinjectdays)
				metrics_by_fpr[fpr]['num_detected'].append(0)

		runtimes.append(runtime)

# general average metrics
runtimes_mean = mean(runtimes)
runtimes_stdev = stdev(runtimes)

# average metrics_by_day
for day in range(options.maxinjectdays):
	for key in ['spatial_precision', 'spatial_recall', 'spatial_overlap', 'topic_precision', 'topic_recall', 'topic_overlap', 'hellinger']:
		metrics_by_day_mean[day][key] = mean(metrics_by_day[day][key])
		metrics_by_day_stdev[day][key] = stdev(metrics_by_day[day][key])

# average metrics_by_fpr
for fpr in range(options.maxbaselinedays):
	for key in ['days_to_detect', 'num_detected']:
		metrics_by_fpr_mean[fpr][key] = mean(metrics_by_fpr[fpr][key])
		metrics_by_fpr_stdev[fpr][key] = stdev(metrics_by_fpr[fpr][key])

results_file = open(results_path, 'w')
results_file.write(str(runtimes_mean) + ',' + str(runtimes_stdev))
results_file.close()

results_by_day_file = open(results_by_day_path, 'w')
for day in range(options.maxinjectdays):
	results_by_day_file.write(
		str(day+1) + ',' +
		str(metrics_by_day_mean[day]['spatial_precision']) + ',' + str(metrics_by_day_mean[day]['spatial_recall']) + ',' + str(metrics_by_day_mean[day]['spatial_overlap']) + ',' +
		str(metrics_by_day_mean[day]['topic_precision']) + ',' + str(metrics_by_day_mean[day]['topic_recall']) + ',' + str(metrics_by_day_mean[day]['topic_overlap']) +  ',' +
		str(metrics_by_day_mean[day]['hellinger']) +  ',' + 
		str(metrics_by_day_stdev[day]['spatial_precision']) + ',' + str(metrics_by_day_stdev[day]['spatial_recall']) + ',' + str(metrics_by_day_stdev[day]['spatial_overlap']) + ',' +
		str(metrics_by_day_stdev[day]['topic_precision']) + ',' + str(metrics_by_day_stdev[day]['topic_recall']) + ',' + str(metrics_by_day_stdev[day]['topic_overlap']) + ',' +
		str(metrics_by_day_stdev[day]['hellinger']) + '\n'
		)
results_by_day_file.close()

results_by_fpr_file = open(results_by_fpr_path, 'w')
for fpr in range(options.maxbaselinedays):
	results_by_fpr_file.write(
		str(fpr+1) + ',' + 
		str(metrics_by_fpr_mean[fpr]['num_detected']) + ',' + str(metrics_by_fpr_mean[fpr]['days_to_detect']) + ',' +
		str(metrics_by_fpr_stdev[fpr]['num_detected']) + ',' + str(metrics_by_fpr_stdev[fpr]['days_to_detect']) + '\n')
results_by_fpr_file.close()
