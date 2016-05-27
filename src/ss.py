# Copyright 2015 Abhinav Maurya

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import fileinput
import logging, json, re
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import operator
import os
import pickle
import random
from scipy.linalg import norm
import scipy.special
import scipy.stats
import time
from copy import deepcopy
from scipy.sparse import csr_matrix
from sklearn.neighbors import KDTree

class SemanticScan:
	def __init__(self):
		"""
		module initialization for file logging
		"""
		logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            filename='log.txt',
            filemode='w')
		self.logger = logging.getLogger('SemanticScan')
		
		# setting timezone for correct reading of string datetimestamps
		os.environ['TZ'] = 'GMT'
		time.tzset()

		# for correctly printing numpy arrays, not useful anywhere anymore
		np.set_printoptions(threshold=np.inf)
		np.seterr(divide='ignore', invalid='ignore')

	def GetEDChiefComplaintCorpus(self, backgrnd_documents_path, foregrnd_documents_path, zipcode_path, stopwords_path=None):
		"""
		read in the ED Chief Complaint corpus and return data structures created for the corpus
		"""
		self.logger.info('Loading the ED Chief Complaint corpus...')

		documents = []
		zipcode_to_geocode = {}
		stopwords = set()

		if stopwords_path is not None:
			for line in fileinput.input(stopwords_path):
				stopwords.update(set(line.lower().strip().split()))

		for line in fileinput.input(zipcode_path):
			line = line.strip().replace('"', '').split(',')
			if len(line) < 5:
				continue
			zipcode = int(line[0])
			state = line[2]
			lat = float(line[3])
			lon = float(line[4])
			#if state == 'PA':
			zipcode_to_geocode[zipcode] = [lat,lon]

		for doc in fileinput.input([backgrnd_documents_path, foregrnd_documents_path]):
			# subsampling to 10% of the documents for quicker testing
			if np.random.binomial(1, 0.1, 1)[0] != 1:
				continue
			doc = doc.lower().strip().split(',')
			if len(doc) < 5 or '' in doc:
				continue
			zipcode = int(doc[1])
			if zipcode not in zipcode_to_geocode:
				continue
			location = zipcode_to_geocode[zipcode]
			words = [re.sub(r'\W+', '', word) for word in doc[2].lower().strip().split()]
			words = [word for word in words if word not in stopwords]
			timestamp = time.mktime(time.strptime(doc[0], '%m/%d/%Y'))
			icd9code = doc[3].split('.')[0]
			injectstatus = int(doc[4])
			if fileinput.filename() in backgrnd_documents_path:
				# background document
				docstatus = 0
			else:
				# foreground document
				docstatus = 1
			documents.append({'words':words, 'timestamp':timestamp, 'location':location, 'zipcode':zipcode, 'icd9code':icd9code, 'docstatus': docstatus, 'injectstatus': injectstatus})

		self.logger.info('Done loading the ED Chief Complaint corpus...')
		return documents

	def GetPnasCorpus(self, documents_path, timestamps_path, stopwords_path, locations_path):
		"""
		read in the PNAS titles corpus and return data structures created for the corpus
		location is empty for PNAS titles corpus
		"""
		self.logger.info('Loading the PNAS corpus...')

		documents = []
		words = []
		timestamps = []
		locations = []
		stopwords = set()

		for line in fileinput.input(stopwords_path):
			stopwords.update(set(line.lower().strip().split()))

		for doc in fileinput.input(documents_path):
			words.append([word for word in doc.lower().strip().split() if word not in stopwords])

		for timestamp in fileinput.input(timestamps_path):
			num_titles = int(timestamp.strip().split()[0])
			timestamp = float(timestamp.strip().split()[1])
			timestamps.extend([timestamp for title in range(num_titles)])

		for location in fileinput.input(locations_path):
			latitude = float(location.strip().split(' ')[0])
			longitude = float(location.strip().split(' ')[1])
			locations.append([latitude, longitude])

		assert len(words) == len(timestamps)
		assert len(words) == len(locations)
		documents = [{'words':words[i], 'timestamp':timestamps[i], 'location':locations[i]} for i in len(words)]

		self.logger.info('Done loading the PNAS corpus...')
		return documents

	def GetTwitterCorpus(self, documents_path, stopwords_path):
		"""
		read in the Twitter corpus and return data structures created for the corpus
		"""
		self.logger.info('Loading the Twitter corpus...')

		documents = []
		stopwords = set()
		prev_timestamp_str = '00/00/0000 00:00:00'
		prev_date_str = '00/00/0000'
		switch_pm = False
		lock_switch_pm = True

		for line in fileinput.input(stopwords_path):
			stopwords.update(set(line.lower().strip().split()))

		for line in fileinput.input(documents_path):
			line = line.strip().split('|')
			if len(line) < 6 or line[1] == '0':
				continue

			latitude = float(line[5])
			longitude = float(line[4])
			if latitude == 0.0 and longitude == 0.0:
				continue
			location = [latitude, longitude]

			line[1] = line[1].replace(' 12:', ' 00:')
			# hit on am to pm change
			if line[1][:10] == prev_timestamp_str[:10] and line[1][-8:] == '00:00:00' and lock_switch_pm == False:
				switch_pm = True
			# hit on date change at midnight
			if line[1][:10] > prev_date_str:																		
				switch_pm = False
				lock_switch_pm = True
			# release lock at 11am to allow am to pm switch at noon
			if line[1][-8:-6] == '11':
				lock_switch_pm = False

			prev_timestamp_str = line[1]
			prev_date_str = line[1][:10]

			doc = line[2]
			words = [word for word in doc.lower().strip().split() if word not in stopwords]
			timestamp = time.mktime(time.strptime(line[1], '%m/%d/%Y %H:%M:%S'))
			if switch_pm:
				timestamp += 43200	# number of seconds in 12 hours

			documents.append({'words':words, 'timestamp':timestamp, 'location':location})

		self.logger.info('Done loading the Twitter corpus...')
		return documents

	def GetYelpCorpus(self, reviews_path, business_path, stopwords_path):
		"""
		read in the Yelp corpus and return data structures created for the corpus
		"""
		self.logger.info('Loading the Yelp corpus...')

		documents = []
		stopwords = set()
		categories = set()
		locations_map = {}

		for line in fileinput.input(stopwords_path):
			stopwords.update(set(line.lower().strip().split()))

		for line in fileinput.input(business_path):
			line = json.loads(line)
			if 'restaurants' in line['categories'] or 'Restaurants' in line['categories']:
				categories_biz = [category.lower().strip() for category in line['categories']]
				categories.update(set(categories_biz))
				locations_map[line['business_id']] = {'latitude':float(line['latitude']), 'longitude':float(line['longitude']), 'state':line['state'], 'city':line['city'], 'categories':categories_biz}

		for line in fileinput.input(reviews_path):
			line = json.loads(line)

			# subsampling to 10% of the documents for quicker testing
			# if np.random.binomial(1, 0.1, 1)[0] != 1:
			#	continue
			if line['business_id'] not in locations_map or locations_map[line['business_id']]['city'] not in ['Las Vegas']:
				continue
			
			words = [re.sub(r'\W+', '', word) for word in line['text'].lower().strip().replace('\n', ' ').split()]	
			words = [word for word in words if word not in stopwords]
			location = [locations_map[line['business_id']]['latitude'], locations_map[line['business_id']]['longitude']]
			timestamp = time.mktime(time.strptime(line['date'], '%Y-%m-%d'))
			
			documents.append({'words':words, 'timestamp':timestamp, 'location':location})
			self.logger.info('  |-Done reading a new line from corpus...')

		self.logger.info('Done loading the Yelp corpus...')
		return documents

	def AddLogRepresentedNumbers(self, num_list):
		"""
		adding very small log-represented numbers in a list
		"""
		if num_list == []:
			return
		result = num_list[0]
		for i in range(1, len(num_list)):
			entry = num_list[i]
			if result < entry:
				result, entry = entry, result
			result += math.log(1 + math.exp(entry - result))
		return result

	"""
	calculate hellinger distance
	"""
	def Hellinger(self, p, q):
		# sqrt(2) with default precision np.float64
		_SQRT2 = np.sqrt(2)
		try:
			return norm(np.sqrt(p) - np.sqrt(q)) / _SQRT2
		except ValueError as e:
			return 4.0

	def CalculateDocumentLikelihood(self, par, d, ghost=0, posterior_estimate=0):
		"""
		calculate document's probability given theta and phi
		"""
		likelihood = 1.0
		num_topics = par['T'] if ghost==0 else par['T_static']

		for i in range(par['N'][d]):
			word = par['w'][d][i]
			likelihood_t = [0.0 for t in range(num_topics)]
			for t in range(num_topics):
				if ghost==0 and posterior_estimate==1:
					likelihood_t[t] = (par['m'][d][t] + par['alpha'][t]) * (par['n'][t][word] + par['beta'][word]) / (par['N'][d] + par['alpha_sum']) / (par['n_sum'][t] + par['beta_sum'])
				if ghost==1 and posterior_estimate==1:
					likelihood_t[t] = (par['m_ghost'][d][t] + par['alpha'][t]) * (par['n'][t][word] + par['beta'][word]) / (par['N'][d] + par['alpha_sum_ghost']) / (par['n_sum'][t] + par['beta_sum'])
				if ghost==0 and posterior_estimate==0 and par['N'][d] > 0 and par['n_sum'][t] > 0:
					likelihood_t[t] = par['m'][d][t] * par['n'][t][word] / par['N'][d] / par['n_sum'][t]
				if ghost==1 and posterior_estimate==0 and par['N'][d] > 0 and par['n_sum'][t] > 0:
					likelihood_t[t] = par['m_ghost'][d][t] * par['n'][t][word] / par['N'][d] / par['n_sum'][t]
			likelihood *= sum(likelihood_t)
		
		return likelihood

	def CalculateStaticCounts(self, par):
		"""
		calculate all counts after assignment of random topics to static document words and before gibbs sampling
		"""
		self.logger.info('  |-Calculating static counts...')

		for d in par['indices_static_doc']:
			for i in range(par['N'][d]):
				topic_di = par['z'][d][i]
				word_di = par['w'][d][i]
				par['m'][d][topic_di] += 1
				par['n'][topic_di][word_di] += 1
				par['n_sum'][topic_di] += 1

		self.logger.info('  |-Done calculating static counts...')

	def CalculateEmergingCounts(self, par):
		"""
		calculate all counts after assignment of random topics to emerging document words and before gibbs sampling
		"""
		self.logger.info('  |-Calculating emerging counts...')

		for d in par['indices_emerging_doc']:
			if par['spatial_compactness_indicator_active'] and not par['doc_spatial_compactness_indicator'][d]:
				continue
			for i in range(par['N'][d]):
				topic_di = par['z'][d][i]
				word_di = par['w'][d][i]
				par['m'][d][topic_di] += 1
				if topic_di in range(par['T_static'], par['T']) or par['options'].static_topics_evolve:
					par['n'][topic_di][word_di] += 1
					par['n_sum'][topic_di] += 1

				# ghost counting
				if par['spatial_compactness_indicator_active']:
					topic_di_ghost = par['z_ghost'][d][i]
					par['m_ghost'][d][topic_di_ghost] += 1

				if par['injectstatus'][d]==1:
					par['injected_topic'][word_di] += 1

		self.logger.info('  |-Done calculating emerging counts...')

	def InitializeStaticParameters(self, documents, options):
		"""
		performing initializations of most parameters
		"""
		self.logger.info('Starting static parameter initialization')

		# constructing vocabulary of all unique words in the documents
		dictionary = set()
		for doc in documents:
			dictionary.update(doc['words'])
		dictionary = list(dictionary)

		# dictionary of all parameters
		par = {}
		# max number of iterations in static gibbs sampling									
		par['max_static_iter'] = options.staticiterations		
		# max number of iterations in emerging gibbs sampling
		par['max_emerging_iter'] = options.emergingiterations	
		# max number of iterations in online document assignment
		par['max_online_iter'] = options.onlineiterations							
		# number of static topics
		par['T_static'] = options.nstatictopics					
		# number of emerging topics
		par['T_emerging'] = options.nemergingtopics
		# total number of topics			
		par['T'] = par['T_static'] + par['T_emerging']
		# total number of documents
		par['D'] = len(documents)
		# total number of unique words in corpus vocabulary
		par['V'] = len(dictionary)
		# total number of words in each document of the corpus
		par['N'] = [len(doc['words']) for doc in documents]
		# options
		par['options'] = options
		# metrics
		par['metrics'] = {}

		# LDA hyperparameters
		par['alpha'] = [1.0/par['T'] for _ in range(par['T'])]
		par['beta'] = [0.1 for _ in range(par['V'])]
		par['alpha_sum'] = sum(par['alpha'])
		par['beta_sum'] = sum(par['beta'])

		# a map from word token to word ID
		par['word_id'] = {dictionary[i]: i for i in range(len(dictionary))}
		# a map from word ID to word token - here just the list of unique words addressable by unique indices
		par['word_token'] = dictionary
		# a map of the corpus from word tokens to word IDs
		par['w'] = [[par['word_id'][documents[d]['words'][i]] for i in range(par['N'][d])] for d in range(par['D'])]
		
		# aggregate counts for collapsed Gibbs sampling
		par['m'] = [[0 for t in range(par['T'])] for d in range(par['D'])]
		par['n'] = [[0 for v in range(par['V'])] for t in range(par['T'])]
		par['n_sum'] = [0 for t in range(par['T'])]

		# injected topics empirical distribution
		par['injected_topic'] = [0 for v in range(par['V'])]

		# set up timestamps (in float format)
		par['t'] = [doc['timestamp'] for doc in documents]
		# set up locations (in [latitude,longitude] format)
		par['l'] = [doc['location'] for doc in documents]
		# set up zipcodes (in int format) - optional, check for KeyError
		try:
			par['zipcodes'] = [doc['zipcode'] for doc in documents]
			par['unique_zipcodes'] = list(set(par['zipcodes']))
		except KeyError as e:
			par['zipcodes'] = None
			par['unique_zipcodes'] = None
		# set up icd9codes (in string format) - optional, check for KeyError
		try:
			par['icd9codes'] = [doc['icd9code'] for doc in documents]
			par['unique_icd9codes'] = list(set(par['icd9codes']))
		except KeyError as e:
			par['icd9codes'] = None
			par['unique_icd9codes'] = None
		# set up docstatus (in int format) - optional, check for KeyError, 0 for background, 1 for foreground
		try:
			par['docstatus'] = [doc['docstatus'] for doc in documents]
		except KeyError as e:
			par['docstatus'] = None
		# set up injectstatus (in int format) - optional, check for KeyError, 0 for non-inject, 1 for inject
		# injects should be found only in foreground documents
		try:
			par['injectstatus'] = [doc['injectstatus'] for doc in documents]
		except KeyError as e:
			par['injectstatus'] = None

		# normalize timestamps and set static and emerging timestamp boundaries
		par['min_timestamp'] = min(par['t'])
		par['max_timestamp'] = max(par['t'])
		if par['docstatus'] is not None and (options.startstatic < 0 or options.endstatic < 0 or options.startemerging < 0 or options.endemerging < 0):
			static_timestamps = [par['t'][d] for d in range(par['D']) if par['docstatus'][d]==0]
			emerging_timestamps = [par['t'][d] for d in range(par['D']) if par['docstatus'][d]==1]
			par['static_first_timestamp'] = min(static_timestamps)
			par['static_second_timestamp'] = max(static_timestamps)
			par['emerging_first_timestamp'] = min(emerging_timestamps)
			par['emerging_second_timestamp'] = max(emerging_timestamps)		
		elif par['docstatus'] is None and (options.startstatic < 0 or options.endstatic < 0 or options.startemerging < 0 or options.endemerging < 0):
			par['static_first_timestamp'] = par['min_timestamp']
			par['static_second_timestamp'] = (par['min_timestamp'] + par['max_timestamp'])/2.0
			par['emerging_first_timestamp'] = par['static_second_timestamp']
			par['emerging_second_timestamp'] = par['max_timestamp']
		else:
			par['static_first_timestamp'] = options.startstatic
			par['static_second_timestamp'] = options.endstatic
			par['emerging_first_timestamp'] = options.startemerging
			par['emerging_second_timestamp'] = options.endemerging
		if options.normalize_timestamps:
			par['t_norm'] = [1.0*(t-par['min_timestamp'])/(par['max_timestamp']-par['min_timestamp']) for t in par['t']]

		# indicator and index lists maintaining which documents form the static and emerging parts of corpus
		par['indicator_static_doc'] = [par['t'][d] >= par['static_first_timestamp'] and par['t'][d] <= par['static_second_timestamp'] for d in range(par['D'])]
		par['indicator_emerging_doc'] = [par['t'][d] > par['emerging_first_timestamp'] and par['t'][d] <= par['emerging_second_timestamp'] for d in range(par['D'])]
		par['indices_static_doc'] = [d for d in range(par['D']) if par['indicator_static_doc'][d]]
		par['indices_emerging_doc'] = [d for d in range(par['D']) if par['indicator_emerging_doc'][d]]

		# random initialization of the topic assignments at all word positions of the corpus
		if par['T_static'] > 0:
			par['z'] = [[random.randrange( 0, par['T_static']*par['indicator_static_doc'][d] + par['T']*(1-par['indicator_static_doc'][d]) ) for _ in range(par['N'][d])] for d in range(par['D'])]
		else:
			par['z'] = [[random.randrange( 0, 1*par['indicator_static_doc'][d] + par['T']*(1-par['indicator_static_doc'][d]) ) for _ in range(par['N'][d])] for d in range(par['D'])]

		# par['simplex_error_indicator_active'] = False 							# indicator whether weak supervision using novelty indicator should be used
		# par['simplex_error_probability'] = [1.0 for _ in range(par['D'])]		# weakly supervised novelty indicator indicating if a document contains an emerging topic

		par['spatial_compactness_indicator_active'] = options.ifspatialcompactness
		par['K'] = options.k

		if par['spatial_compactness_indicator_active']:
			# sparsity, alpha_sum_ghost, zipcode_subsumes_docs are found only in first initialization not elsewhere e.g. not found in ReinitializeEmergingParameters()
			par['sparsity'] = options.sparsity
			par['alpha_sum_ghost'] = sum(par['alpha'][:par['T_static']])

			par['zipcode_spatial_compactness_indicator'] = {z:1 for z in par['unique_zipcodes']}
			par['doc_spatial_compactness_indicator'] = [1 for d in range(par['D'])]

			self.logger.info('  |-Computing KD-Tree and K Nearest Neighbors for emerging documents...')
			par['kdt'] = KDTree(np.array([par['l'][d] for d in range(len(par['l'])) if par['indicator_emerging_doc'][d]]), leaf_size=30, metric='euclidean')
			par['knn'] = par['kdt'].query(np.array(par['l']), k=par['K'], return_distance=False)
			self.logger.info('  |-Done computing KD-Tree and K Nearest Neighbors for emerging documents...')

			self.logger.info('  |-Assigning ghost variables for finding likelihood of new documents with static topics...')
			par['m_ghost'] = [[0 for t in range(par['T_static'])] for d in range(par['D'])]
			par['z_ghost'] = [[random.randrange(0, par['T_static']) for _ in range(par['N'][d])] for d in range(par['D'])]
			self.logger.info('  |-Done assigning ghost variables for finding likelihood of new documents with static topics...')
		else:
			self.logger.info('  |-Spatial compactness flag off.')
			self.logger.info('  |-Skipping computation of KD-Tree and K Nearest Neighbors for emerging documents...')
			self.logger.info('  |-Skipping assignment of ghost variables for finding likelihood of new documents with static topic only...')

		par['spatial_scan_active'] = options.ifspatialscan
		par['document_assignment'] = [0 for d in range(par['D'])]
		par['start_day'] = -1
		par['inject_start_day'] = -1
		par['inject_end_day'] = -1
		par['window_days'] = options.windowdays
		par['baseline_days'] = options.baselinedays
		par['heldout_icd9code'] = [par['icd9codes'][d] for d in range(par['D']) if par['injectstatus'][d]==1][0]

		# do the aggregate counts due to the z assignments
		self.CalculateStaticCounts(par)

		self.logger.info('Done with static parameter initialization')
		return par

	def InitializeEmergingParameters(self, par):
		"""
		initializing model counts before emerging topic estimation
		"""
		self.logger.info('Starting emerging parameter initialization')
		self.CalculateEmergingCounts(par)
		self.logger.info('Done with emerging parameter initialization')
		return par

	def ReinitializeEmergingParameters(self, par, start_day=-1):
		"""
		reinitializing topic assignments in emerging documents to estimate the new topics afresh after spatial compactness probability estimation
		start_day: zero-indexed date on which emerging doc window starts 
		"""
		self.logger.info('Starting emerging parameter reinitialization')

		emerging_timestamps = [par['t'][d] for d in range(par['D']) if par['docstatus'][d]==1]
		unique_emerging_timestamps = list(sorted(list(set(emerging_timestamps))))
		unique_emerging_timestamp_differences = [unique_emerging_timestamps[i+1]-unique_emerging_timestamps[i] for i in range(len(unique_emerging_timestamps)-1)]
		if_unique_emerging_timestamps_arent_days = [int((int(unique_emerging_timestamp_differences[i])%86400)!=0) for i in range(len(unique_emerging_timestamp_differences))]
		assert sum(if_unique_emerging_timestamps_arent_days)==0

		par['start_day'] = start_day
		timestamps_of_injected_documents = [par['t'][d] for d in range(par['D']) if par['injectstatus'][d]==1]
		par['inject_start_day'] = unique_emerging_timestamps.index(min(timestamps_of_injected_documents))
		par['inject_end_day'] = unique_emerging_timestamps.index(max(timestamps_of_injected_documents))

		# random initialization of the topic assignments at all word positions of the corpus
		# we only care about this for emerging docs but we will just go ahead and do it for all documents
		# if par['T_static'] > 0:
		#	par['z'] = [[random.randrange( 0, par['T_static']*par['indicator_static_doc'][d] + par['T']*(1-par['indicator_static_doc'][d]) ) for _ in range(par['N'][d])] for d in range(par['D'])]
		# else:
		#	par['z'] = [[random.randrange( 0, 1*par['indicator_static_doc'][d] + par['T']*(1-par['indicator_static_doc'][d]) ) for _ in range(par['N'][d])] for d in range(par['D'])]

		# aggregate counts for collapsed Gibbs sampling
		par['m'] = [[0 for t in range(par['T'])] for d in range(par['D'])]
		par['n'][par['T_static']:par['T']] = [[0 for v in range(par['V'])] for t in range(par['T_static'], par['T'])]
		par['n_sum'][par['T_static']:par['T']] = [0 for t in range(par['T_static'], par['T'])]
		par['injected_topic'] = [0 for v in range(par['V'])]

		# changing the set of emerging documents considered for reinitialization by using the day parameter
		if start_day >= 0:
			self.logger.info('  |-A particular start day specified. Need to recalculate the set of baseline and emerging document indices...')
			self.logger.info('  |-Reassigning emerging documents using start day {0} and window length of {1} days...'.format(start_day, par['window_days']))

			# all unique timestamps after 2003 i.e. from 2004 only
			assert len(unique_emerging_timestamps) >= start_day+par['window_days']
			assert start_day >= par['baseline_days']

			allowed_emerging_timestamps = unique_emerging_timestamps[start_day:start_day+par['window_days']]
			min_allowed_emerging_timestamps = min(allowed_emerging_timestamps)
			max_allowed_emerging_timestamps = max(allowed_emerging_timestamps)
			par['indicator_emerging_doc'] = [par['t'][d] >= min_allowed_emerging_timestamps and par['t'][d] <= max_allowed_emerging_timestamps for d in range(par['D'])]
			par['indices_emerging_doc'] = [d for d in range(par['D']) if par['indicator_emerging_doc'][d]]

			# we need baseline document indices only when performing spatial scan
			if par['spatial_scan_active']:
				allowed_baseline_timestamps = unique_emerging_timestamps[start_day-par['baseline_days']:start_day]
				min_allowed_baseline_timestamps = min(allowed_baseline_timestamps)
				max_allowed_baseline_timestamps = max(allowed_baseline_timestamps)
				par['indicator_baseline_doc'] = [par['t'][d] >= min_allowed_baseline_timestamps and par['t'][d] <= max_allowed_baseline_timestamps for d in range(par['D'])]
				par['indices_baseline_doc'] = [d for d in range(par['D']) if par['indicator_baseline_doc'][d]]

			# reevaluating kdtree since the emerging document indices have changed
			# exactly the same as in the function InitializeStaticParameters()
			if par['spatial_compactness_indicator_active']:
				par['zipcode_spatial_compactness_indicator'] = {z:1 for z in par['unique_zipcodes']}
				par['doc_spatial_compactness_indicator'] = [1 for d in range(par['D'])]
			
				self.logger.info('  |-Computing KD-Tree and K Nearest Neighbors for emerging documents...')
				par['kdt'] = KDTree(np.array([par['l'][d] for d in range(len(par['l'])) if par['indicator_emerging_doc'][d]]), leaf_size=30, metric='euclidean')
				par['knn'] = par['kdt'].query(np.array(par['l']), k=par['K'], return_distance=False)
				self.logger.info('  |-Done computing KD-Tree and K Nearest Neighbors for emerging documents...')

				self.logger.info('  |-Assigning ghost variables for finding likelihood of new documents with static topics...')
				par['m_ghost'] = [[0 for t in range(par['T_static'])] for d in range(par['D'])]
				par['z_ghost'] = [[random.randrange(0, par['T_static']) for _ in range(par['N'][d])] for d in range(par['D'])]
				self.logger.info('  |-Done assigning ghost variables for finding likelihood of new documents with static topics...')
			else:
				self.logger.info('  |-Spatial compactness flag off.')
				self.logger.info('  |-Skipping computation of KD-Tree and K Nearest Neighbors for emerging documents...')
				self.logger.info('  |-Skipping assignment of ghost variables for finding likelihood of new documents with static topic only...')

		# redo the aggregate counts due to the new z assignments
		self.CalculateEmergingCounts(par)

		self.logger.info('Done with emerging parameter reinitialization')
		return par

	def CalculateSimplexErrorProbability(self, par):
		"""
		assigns a probability to each emerging document of being included in the emerging topic estimation
		called only once before emerging topic estimation to calculate error of explaining emerging documents with just static topics
		"""
		if not par['simplex_error_indicator_active']:
			return
		self.logger.info('Starting calculation of novelty indicators...')

		phi = self.ComputePosteriorEstimateOfPhi(par).getT()[:,0:par['T_static']]
		phi_transpose = phi.getT()
		sparse_phi = csr_matrix(phi)
		projection_operator = np.dot(np.dot(phi_transpose, phi).getI(), phi_transpose)
		projection_operator = csr_matrix(projection_operator)

		for d in par['indices_emerging_doc']:
			doc_vector = np.matrix([0 for word_id in range(par['V'])]).getT()
			for i in par['w'][d]:
				doc_vector[i] += 1
			doc_coordinates = projection_operator.dot(doc_vector)
			doc_projection = sparse_phi.dot(doc_coordinates)
			doc_error = doc_vector - doc_projection
			doc_error_norm = np.linalg.norm(doc_error)/np.linalg.norm(doc_vector)
			par['simplex_error_probability'][d] = min(1.0, doc_error_norm)
			par['simplex_error_probability'][d] = np.random.binomial(1, par['simplex_error_probability'][d], 1)[0]

		self.logger.info('Ending calculation of novelty indicators...')

	def CalculateDocumentAssignment(self, par, include_static=False):
		"""
		assign documents to static or emerging topics
		"""
		self.logger.info('Starting document to topic assignment for spatial scan')

		phi = self.ComputePosteriorEstimateOfPhi(par)

		if include_static or par['T_emerging']==0:
			num_topics = par['T']
		else:
			num_topics = par['T_emerging']
		
		for d in par['indices_baseline_doc'] + par['indices_emerging_doc']:
			theta_d = [1.0/num_topics for _ in range(num_topics)]

			for itr in range(par['max_online_iter']):
				offset = par['T'] - num_topics
				pr_z = [[ phi[ offset+k, par['w'][d][i] ] * theta_d[k] for k in range(num_topics) ] for i in range(par['N'][d])]
				sum_pr_z = [sum(pr_z[i]) for i in range(par['N'][d])]
				pr_z = [[1.0*pr_z[i][k]/sum_pr_z[i] for k in range(num_topics)] for i in range(par['N'][d])]
				theta_d = [1.0/num_topics + sum([pr_z[i][k] for i in range(par['N'][d])]) for k in range(num_topics)]

			theta_d = np.asarray(theta_d)
			par['document_assignment'][d] = np.argmax(theta_d)
			if d%1000==0:
				self.logger.info('  |-Calculating topic assignment for document {document}...'.format(document=d))
			
		self.logger.info('Ending document to topic assignment for spatial scan')
		return par

	def PerformSpatialScan(self, par, include_static=False):
		"""
		perform circular spatial scan using results of topic modeling and online document assignment
		scan_type: 0=spatially constrained LTSS, 1=circular spatial scan, 2=naive count-based scan
		"""
		if not par['spatial_scan_active']:
			self.logger.info('Trying to perform spatial scan without the spatial scan flag turned on!')
			self.logger.info('Skipping spatial scan. If you want to perform spatial scan, turn on the ifspatialscan flag.')
			return par

		if par['options'].scan_type == 2:
			return self.PerformNaiveSpatialScan(par)

		self.logger.info('Starting spatial scan...')

		if include_static or par['T_emerging']==0:
			num_topics = par['T']
		else:
			num_topics = par['T_emerging']

		zipcode_to_geocode = {par['zipcodes'][i]:par['l'][i] for i in range(par['D'])}
		unique_zipcodes = list(zipcode_to_geocode.keys())
		unique_locations = list(zipcode_to_geocode.values())
		num_zipcodes = len(unique_zipcodes)
		
		par['spatial_scan_counts'] = {u:[0 for t in range(num_topics)] for u in unique_zipcodes}
		par['spatial_scan_baselines'] = {u:[0 for t in range(num_topics)] for u in unique_zipcodes}
		par['spatial_scan_scores'] = {u:[[0 for t in range(num_topics)] for k in range(par['K'])] for u in unique_zipcodes}

		for d in par['indices_emerging_doc']:
			zipcode = par['zipcodes'][d]
			assigned_topic = par['document_assignment'][d]
			par['spatial_scan_counts'][zipcode][assigned_topic] += 1

		for d in par['indices_baseline_doc']:
			zipcode = par['zipcodes'][d]
			assigned_topic = par['document_assignment'][d]
			par['spatial_scan_baselines'][zipcode][assigned_topic] += 1

		kdt_results = KDTree(np.array(unique_locations), leaf_size=30, metric='euclidean')
		knn_results = kdt_results.query(np.array(unique_locations), k=par['K'], return_distance=False)

		par['max_score_location'] = {'score':0.0, 'zipcode':-1, 'k':-1, 't':-1, 'knn_list':[], 'b':1, 'c':1}
		for i in range(num_zipcodes):
			zipcode = unique_zipcodes[i]
			for t in range(num_topics):
				b, c = 0, 0
				knn_list = []
				priority_list = []
				
				for k in range(par['K']):
					knn = unique_zipcodes[knn_results[i][k]]
					knn_list.append(knn)
					b += par['spatial_scan_baselines'][knn][t]
					c += par['spatial_scan_counts'][knn][t]
					priority_list.append([b,c,knn])

					score = self.CalculateExpectationBasedPoissonScore(1.0*b*par['window_days']/par['baseline_days'], c)
					if par['max_score_location']['score'] <= score:
						par['max_score_location'] = {'score':score, 'zipcode':zipcode, 'k':k, 't':t, 'knn_list':knn_list[:], 'b':b, 'c':c}

				if par['options'].scan_type==0:		
					b, c = 1, 1
					knn_list = []
					priority_list = sorted(priority_list, key=lambda x: x[1]/x[0] if x[0]>0 else 0.0, reverse=True)
					
					for k in range(len(priority_list)):
						knn_list.append(priority_list[k][2])
						b += priority_list[k][0]
						c += priority_list[k][1]
						
						score = self.CalculateExpectationBasedPoissonScore(1.0*b*par['window_days']/par['baseline_days'], c)
						if par['max_score_location']['score'] <= score:
							par['max_score_location'] = {'score':score, 'zipcode':zipcode, 'k':k, 't':t, 'knn_list':knn_list[:], 'b':b, 'c':c}

		self.logger.info('  |-Found a max scoring neighborhood using circular spatial scan: ' + str(par['max_score_location']))

		phi = self.ComputePosteriorEstimateOfPhi(par)
		detected_topic = phi[par['T'] - num_topics + par['max_score_location']['t'],:]
		injected_topic = np.asarray(par['injected_topic'])
		injected_topic = injected_topic/sum(injected_topic)
		par['max_score_location']['hellinger'] = self.Hellinger(detected_topic, injected_topic)

		# overwrite these indictor lists only if they are not being used by spatial compactness methods 
		# like CalculateSpatialCompactnessProbability() or InferSpatialCompactnessUsingGFSS()
		if not par['spatial_compactness_indicator_active']:
			par['zipcode_spatial_compactness_indicator'] = {z:1*(z in par['max_score_location']['knn_list']) for z in par['unique_zipcodes']}
			par['doc_spatial_compactness_indicator'] = [1 if par['zipcodes'][d] in par['max_score_location']['knn_list'] else 0 for d in range(par['D'])]

		self.logger.info('Finished spatial scan...')
		return par

	def PerformNaiveSpatialScan(self, par, include_static=False):
		"""
		perform naive spatial scan
		"""
		if not par['spatial_scan_active']:
			self.logger.info('Trying to perform spatial scan without the spatial scan flag turned on!')
			self.logger.info('Skipping spatial scan. If you want to perform spatial scan, turn on the ifspatialscan flag.')
			return par

		self.logger.info('Starting spatial scan...')

		if include_static or par['T_emerging']==0:
			num_topics = par['T']
		else:
			num_topics = par['T_emerging']

		zipcode_to_geocode = {par['zipcodes'][i]:par['l'][i] for i in range(par['D'])}
		unique_zipcodes = list(zipcode_to_geocode.keys())
		unique_locations = list(zipcode_to_geocode.values())
		num_zipcodes = len(unique_zipcodes)
		
		par['spatial_scan_counts'] = {u:[0 for t in range(par['T'])] for u in unique_zipcodes}
		par['spatial_scan_baselines'] = {u:[0 for t in range(par['T'])] for u in unique_zipcodes}
		par['spatial_scan_scores'] = {u:[[0 for t in range(par['T'])] for k in range(par['K'])] for u in unique_zipcodes}
		par['spatial_scan_topic_aggregate_counts'] = [0 for t in range(par['T'])]

		for d in par['indices_emerging_doc']:
			zipcode = par['zipcodes'][d]
			assigned_topic = par['document_assignment'][d]
			par['spatial_scan_counts'][zipcode][assigned_topic] += 1
			par['spatial_scan_topic_aggregate_counts'][assigned_topic] += 1

		max_topic_count = max(par['spatial_scan_topic_aggregate_counts'])
		detected_topic = par['spatial_scan_topic_aggregate_counts'].index(max_topic_count)
		detected_topic_zipcode_counts = {u:par['spatial_scan_counts'][u][detected_topic] for u in unique_zipcodes}
		detected_topic_sorted_zipcode_counts = sorted(detected_topic_zipcode_counts.items(), key=operator.itemgetter(1), reverse=True)
		detected_zipcodes = [detected_topic_sorted_zipcode_counts[k][0] for k in range(par['K'])]
		detected_zipcodes_score = sum([detected_topic_sorted_zipcode_counts[k][1] for k in range(par['K'])])

		par['max_score_location'] = {'score':max_topic_count, 'zipcode':-1, 'k':par['K'], 't':detected_topic, 'knn_list':detected_zipcodes, 'c':detected_zipcodes_score}

		self.logger.info('  |-Found a max scoring neighborhood using circular spatial scan: ' + str(par['max_score_location']))

		phi = self.ComputePosteriorEstimateOfPhi(par)
		detected_topic_ = phi[par['T'] - num_topics + par['max_score_location']['t'],:]
		injected_topic_ = np.asarray(par['injected_topic'])
		injected_topic_ = injected_topic_/sum(injected_topic_)
		par['max_score_location']['hellinger'] = self.Hellinger(detected_topic_, injected_topic_)

		par['zipcode_spatial_compactness_indicator'] = {z:1*(z in par['max_score_location']['knn_list']) for z in par['unique_zipcodes']}
		par['doc_spatial_compactness_indicator'] = [1 if par['document_assignment'][d]==detected_topic else 0 for d in range(par['D'])]
		#par['doc_spatial_compactness_indicator'] = [1 if par['zipcodes'][d] in par['max_score_location']['knn_list'] else 0 for d in range(par['D'])]

		self.logger.info('Finished spatial scan...')
		return par

	def CalculateExpectationBasedPoissonScore(self, b, c):
		if c <= b or b==0:
			return 0.0
		else:
			return c*math.log(1.0*c/b) + b - c

	def CalculateSpatialCompactnessProbability(self, par):
		"""
		assigns a probability to each emerging document of being included in the emerging topic estimation
		called after every emerging topic gibbs sampling iteration to iteratively find spatially localized emerging topics
		"""
		if not par['spatial_compactness_indicator_active']:
			self.logger.info('Spatial compactness flag off. Skipping inference of spatial compactness...')
			return
		if par['T_static'] == 0:
			self.logger.info('Number of static topics is 0. Skipping inference of spatial compactness...')
			return

		self.logger.info('Starting inference of spatial compactness...')

		# setting up huge local lists required for posterior probability calculations
		likelihood_static_topics = [0.0 for _ in range(par['D'])]
		likelihood_all_topics = [0.0 for _ in range(par['D'])]
		smoothed_likelihood_ratios = [0.0 for _ in range(par['D'])]

		# setting up a dictionary to avoid nasty space guzzling
		prob_topic_d_k = {d:{} for d in range(par['D'])}

		for d in par['indices_emerging_doc']:
			likelihood_all_topics[d] = self.CalculateDocumentLikelihood(par, d)
			likelihood_static_topics[d] = self.CalculateDocumentLikelihood(par, d, ghost=1)
			if likelihood_static_topics[d] == 0.0:
				likelihood_ratio = 100.0
			else:
				likelihood_ratio = likelihood_all_topics[d] / likelihood_static_topics[d]
			smoothed_likelihood_ratios[d] = 1 - par['sparsity'] + par['sparsity'] * likelihood_ratio
			if d%1000 == 0:
				self.logger.info('  |-Calculating likelihood ratio for document {document}...'.format(document=d))
		likelihood_static_topics = []
		likelihood_all_topics = []

		self.logger.info('  |-Starting calculation of posterior probabilities of finding novel topics in documents...')

		self.logger.info('    |-Calculating probability of new topic for each document d and each neighborhood size k...')
		for d in par['indices_emerging_doc']:
			log_cumprod_smoothed_likelihood_ratios = 0.0
			for k in range(par['K']):
				knn = par['indices_emerging_doc'][par['knn'][d][k]]
				log_cumprod_smoothed_likelihood_ratios += math.log(smoothed_likelihood_ratios[knn])
				prob_topic_d_k[d][k] = log_cumprod_smoothed_likelihood_ratios
		smoothed_likelihood_ratios = []

		self.logger.info('    |-Normalizing calculated probabilities over all d and k...')
		normalizer_ = self.AddLogRepresentedNumbers([self.AddLogRepresentedNumbers(list(prob_topic_d_k[d].values())) for d in par['indices_emerging_doc']])
		# do not compress this double for loop!
		for d in par['indices_emerging_doc']:
			for k in range(par['K']):
				prob_topic_d_k[d][k] = math.exp(prob_topic_d_k[d][k] - normalizer_)

		self.logger.info('    |-Calculating posterior probability of new topic for each document d...')

		par['doc_spatial_compactness_probability'] = [0.0 for d in range(par['D'])]
		par['doc_spatial_compactness_indicator'] = [0 for d in range(par['D'])]
		for d in par['indices_emerging_doc']:
			cumsum_prob_topic_d_k = 0.0
			for k in reversed(range(par['K'])):
				knn = par['indices_emerging_doc'][par['knn'][d][k]]
				cumsum_prob_topic_d_k += prob_topic_d_k[d][k]
				par['doc_spatial_compactness_probability'][knn] += cumsum_prob_topic_d_k
				par['doc_spatial_compactness_probability'][knn] = min(1.0, par['doc_spatial_compactness_probability'][knn])

		for d in par['indices_emerging_doc']:
			par['doc_spatial_compactness_indicator'][d] = np.random.binomial(1, par['doc_spatial_compactness_probability'][d], 1)[0]

		self.logger.info('  |-Ending calculation of posterior probabilities of finding novel topics in documents...')

		self.logger.info('Ending inference of spatial compactness...')

	def InferSpatialCompactnessUsingGFSS(self, par):
		"""
		assigns a probability to each emerging document of being included in the emerging topic estimation
		called after every emerging topic gibbs sampling iteration to iteratively find spatially localized emerging topics
		functionally identical to CalculateSpatialCompactnessProbability() but performs GFSS at zipcode level instead of document level
		"""
		if not par['spatial_compactness_indicator_active']:
			self.logger.info('Spatial compactness flag off. Skipping inference of spatial compactness...')
			return
		if par['T_static'] == 0:
			self.logger.info('Number of static topics is 0. Skipping inference of spatial compactness...')
			return

		self.logger.info('Starting inference of spatial compactness...')

		# setting up zipcode stuff
		zipcode_to_geocode = {par['zipcodes'][d]:par['l'][d] for d in par['indices_emerging_doc']}
		unique_zipcodes = list(zipcode_to_geocode.keys())
		unique_locations = list(zipcode_to_geocode.values())
		num_zipcodes = len(unique_zipcodes)

		zipcode_to_geocode_all = {par['zipcodes'][d]:par['l'][d] for d in range(par['D'])}
		unique_zipcodes_all = list(zipcode_to_geocode_all.keys())
		unique_locations_all = list(zipcode_to_geocode_all.values())
		num_zipcodes_all = len(unique_zipcodes_all)

		# setting up huge local lists required for posterior probability calculations
		likelihood_static_topics = [0.0 for _ in range(par['D'])]
		likelihood_all_topics = [0.0 for _ in range(par['D'])]
		doc_likelihood_ratios = [0.0 for _ in range(par['D'])]
		smoothed_likelihood_ratios = {z:1.0 for z in unique_zipcodes}

		for d in par['indices_emerging_doc']:
			zipcode = par['zipcodes'][d]
			likelihood_all_topics[d] = self.CalculateDocumentLikelihood(par, d)
			likelihood_static_topics[d] = self.CalculateDocumentLikelihood(par, d, ghost=1)
			if likelihood_static_topics[d] == 0.0:
				likelihood_ratio = 100.0
			else:
				likelihood_ratio = likelihood_all_topics[d] / likelihood_static_topics[d]
			doc_likelihood_ratios[d] = likelihood_ratio
			smoothed_likelihood_ratios[zipcode] *= doc_likelihood_ratios[d]
			if d%1000 == 0:
				self.logger.info('  |-Calculating likelihood ratio for document {document}...'.format(document=d))

		# setting up KDT using locations stuff
		kdt_results = KDTree(np.array(unique_locations), leaf_size=30, metric='euclidean')
		knn_results = kdt_results.query(np.array(unique_locations), k=par['K'], return_distance=False)

		# normalize so that zipcodes with larger number of documents are not penalized with lower log likelihood
		for z in unique_zipcodes:
			smoothed_likelihood_ratios[z] = 1 - par['sparsity'] + par['sparsity'] * smoothed_likelihood_ratios[z]

		self.logger.info('  |-Starting calculation of posterior probabilities of finding novel topics in documents...')

		# setting up a dictionary to avoid nasty space guzzling
		prob_topic_z_k = {z:{} for z in unique_zipcodes}

		self.logger.info('    |-Calculating probability of new topic for each document d and each neighborhood size k...')
		for i in range(len(unique_zipcodes)):
			z = unique_zipcodes[i]
			log_cumprod_smoothed_likelihood_ratios = 0.0
			for k in range(par['K']):
				knn = unique_zipcodes[knn_results[i][k]]
				log_cumprod_smoothed_likelihood_ratios += math.log(smoothed_likelihood_ratios[knn])
				prob_topic_z_k[z][k] = log_cumprod_smoothed_likelihood_ratios

		self.logger.info('    |-Normalizing calculated probabilities over all d and k...')
		normalizer_ = self.AddLogRepresentedNumbers([self.AddLogRepresentedNumbers(list(prob_topic_z_k[z].values())) for z in unique_zipcodes])
		# do not compress this double for loop!
		for z in unique_zipcodes:
			for k in range(par['K']):
				prob_topic_z_k[z][k] = math.exp(prob_topic_z_k[z][k] - normalizer_)

		self.logger.info('    |-Calculating posterior probability of new topic for each document d...')
		par['zipcode_spatial_compactness_probability'] = {z:0.0 for z in unique_zipcodes_all}
		par['zipcode_spatial_compactness_indicator'] = {z:0 for z in unique_zipcodes_all}
		par['doc_spatial_compactness_probability'] = [0.0 for d in range(par['D'])]
		par['doc_spatial_compactness_indicator'] = [0 for d in range(par['D'])]
		for i in range(len(unique_zipcodes)):
			z = unique_zipcodes[i]
			cumsum_prob_topic_z_k = 0.0
			for k in reversed(range(par['K'])):
				knn = unique_zipcodes[knn_results[i][k]]
				cumsum_prob_topic_z_k += prob_topic_z_k[z][k]
				par['zipcode_spatial_compactness_probability'][knn] += cumsum_prob_topic_z_k
				par['zipcode_spatial_compactness_probability'][knn] = min(1.0, par['zipcode_spatial_compactness_probability'][knn])

		idx = math.floor(par['K']*par['sparsity'])
		threshold = list(sorted(list(par['zipcode_spatial_compactness_probability'].values()), reverse=True))[idx]
		for zipcode in unique_zipcodes:
			par['zipcode_spatial_compactness_indicator'][zipcode] = 1 if par['zipcode_spatial_compactness_probability'][zipcode]>=threshold else 0
		# for zipcode in unique_zipcodes_all:
		# 	par['zipcode_spatial_compactness_indicator'][zipcode] = np.random.binomial(1, par['zipcode_spatial_compactness_probability'][zipcode], 1)[0]
		
		for d in par['indices_emerging_doc']:
			zipcode = par['zipcodes'][d]
			par['doc_spatial_compactness_probability'][d] = par['zipcode_spatial_compactness_probability'][zipcode]
			par['doc_spatial_compactness_indicator'][d] = par['zipcode_spatial_compactness_indicator'][zipcode]
			# par['doc_spatial_compactness_indicator'][d] = np.random.binomial(1, par['zipcode_spatial_compactness_probability'][zipcode], 1)[0]

		self.logger.info('  |-Ending calculation of posterior probabilities of finding novel topics in documents...')

		self.logger.info('Ending inference of spatial compactness...')

	def ComputePosteriorEstimatesOfThetaAndPhi(self, par):
		"""
		computing mle estimates of theta and phi
		"""
		theta = deepcopy(par['m'])
		phi = deepcopy(par['n'])
		alpha = np.asarray(par['alpha'])
		beta = np.asarray(par['beta'])

		for d in range(par['D']):
			if sum(theta[d]) == 0:
				theta[d] = np.asarray([1.0/len(theta[d]) for _ in range(len(theta[d]))])
			else:
				theta[d] = np.asarray(theta[d])
				theta[d] = 1.0 * (theta[d] + alpha) / (sum(theta[d]) + sum(alpha))
		theta = np.asarray(theta)

		for t in range(par['T']):
			if sum(phi[t]) == 0:
				phi[t] = np.asarray([1.0/len(phi[t]) for _ in range(len(phi[t]))])
			else:
				phi[t] = np.asarray(phi[t])
				phi[t] = 1.0 * (phi[t] + beta) / (sum(phi[t]) + sum(beta))
		phi = np.asarray(phi)

		return theta, phi

	def ComputePosteriorEstimatesOfTheta(self, par, ghost=0):
		"""
		computing mle estimates of theta
		"""
		if ghost:
			theta = deepcopy(par['m_ghost'])
		else:
			theta = deepcopy(par['m'])
		theta_size = len(theta[0])
		alpha = np.asarray(par['alpha'][:theta_size])

		for d in range(par['D']):
			if sum(theta[d]) == 0:
				theta[d] = np.asarray([1.0/len(theta[d]) for _ in range(len(theta[d]))])
			else:
				theta[d] = np.asarray(theta[d])
				theta[d] = 1.0 * (theta[d] + alpha) / (sum(theta[d]) + sum(alpha))
		theta = np.asarray(theta)

		return theta

	def ComputePosteriorEstimateOfPhi(self, par):
		"""
		computing mle estimates of phi
		"""
		phi = deepcopy(par['n'])
		beta = np.asarray(par['beta'])

		for t in range(par['T']):
			if sum(phi[t]) == 0:
				phi[t] = np.asarray([1.0/len(phi[t]) for _ in range(len(phi[t]))])
			else:
				phi[t] = np.asarray(phi[t])
				phi[t] = 1.0 * (phi[t] + beta) / (sum(phi[t]) + sum(beta))
		phi = np.asarray(phi)

		return phi

	def GetMetrics(self, par):
		"""
		calculating metrics after SCSS
		"""
		par['metrics'].update({'heldout_icd9code':par['heldout_icd9code'], 'start_day':par['start_day'],
							'inject_start_day':par['inject_start_day'], 'inject_end_day':par['inject_end_day'], 
							'window_days':par['window_days'], 'baseline_days':par['baseline_days'],
							'score':par['max_score_location']['score'], 'hellinger':par['max_score_location']['hellinger']})
		max_topic = par['max_score_location']['t']

		# get all infected zipcodes
		# true_infected_zipcodes = set([par['zipcodes'][d] for d in par['indices_emerging_doc'] if par['injectstatus'][d]==1])
		true_infected_zipcodes = set([par['zipcodes'][d] for d in range(par['D']) if par['injectstatus'][d]==1])
		estimated_infected_zipcodes = set([z for z in par['unique_zipcodes'] if par['zipcode_spatial_compactness_indicator'][z]==1])
		zipcode_intersection = true_infected_zipcodes.intersection(estimated_infected_zipcodes)
		zipcode_union = true_infected_zipcodes.union(estimated_infected_zipcodes)

		# get all infected docs by SS
		true_infected_docs = set([d for d in par['indices_emerging_doc'] if par['injectstatus'][d]==1])
		estimated_infected_docs_by_ss = set([d for d in par['indices_emerging_doc'] if par['doc_spatial_compactness_indicator'][d]==1])
		doc_intersection_by_ss = true_infected_docs.intersection(estimated_infected_docs_by_ss)
		doc_union_by_ss = true_infected_docs.union(estimated_infected_docs_by_ss)

		# get all infected docs by topic
		true_infected_docs = set([d for d in par['indices_emerging_doc'] if par['injectstatus'][d]==1])
		estimated_infected_docs_by_topic_assignment = set([d for d in par['indices_emerging_doc'] if par['document_assignment'][d]==max_topic])
		doc_intersection_by_topic_assignment = true_infected_docs.intersection(estimated_infected_docs_by_topic_assignment)
		doc_union_by_topic_assignment = true_infected_docs.union(estimated_infected_docs_by_topic_assignment)

		inside_window = par['start_day'] + par['window_days'] - 1 >= par['inject_start_day'] and par['start_day'] <= par['inject_end_day']

		par['metrics']['spatial_precision'] = 1.0*len(zipcode_intersection)/len(estimated_infected_zipcodes) if len(estimated_infected_zipcodes) > 0 and inside_window else 0.0
		par['metrics']['spatial_recall'] = 1.0*len(zipcode_intersection)/len(true_infected_zipcodes) if len(true_infected_zipcodes) > 0 and inside_window else 0.0
		par['metrics']['spatial_overlap'] = 1.0*len(zipcode_intersection)/len(zipcode_union) if len(zipcode_union) > 0 and inside_window else 0.0

		par['metrics']['doc_precision'] = 1.0*len(doc_intersection_by_ss)/len(estimated_infected_docs_by_ss) if len(estimated_infected_docs_by_ss) > 0 and inside_window else 0.0
		par['metrics']['doc_recall'] = 1.0*len(doc_intersection_by_ss)/len(true_infected_docs) if len(true_infected_docs) > 0 and inside_window else 0.0
		par['metrics']['doc_overlap'] = 1.0*len(doc_intersection_by_ss)/len(doc_union_by_ss) if len(doc_union_by_ss) > 0 and inside_window else 0.0

		par['metrics']['topic_precision'] = 1.0*len(doc_intersection_by_topic_assignment)/len(estimated_infected_docs_by_topic_assignment) if len(estimated_infected_docs_by_topic_assignment) > 0 and inside_window else 0.0
		par['metrics']['topic_recall'] = 1.0*len(doc_intersection_by_topic_assignment)/len(true_infected_docs) if len(true_infected_docs) > 0 and inside_window else 0.0
		par['metrics']['topic_overlap'] = 1.0*len(doc_intersection_by_topic_assignment)/len(doc_union_by_topic_assignment) if len(doc_union_by_topic_assignment) > 0 and inside_window else 0.0

		print(str(par['metrics']['spatial_precision']) + ',' + str(par['metrics']['spatial_recall']) + ',' + str(par['metrics']['spatial_overlap']) + ',' + 
			str(par['metrics']['doc_precision']) + ',' + str(par['metrics']['doc_recall']) + ',' + str(par['metrics']['doc_overlap']) + ',' + 
			str(par['metrics']['topic_precision']) + ',' + str(par['metrics']['topic_recall']) + ',' + str(par['metrics']['topic_overlap']) + ',' + 
			str(par['metrics']['heldout_icd9code']) + ',' + str(par['metrics']['start_day']) + ',' + str(par['metrics']['inject_start_day']) + ',' + 
			str(par['metrics']['inject_end_day']) + ',' + str(par['metrics']['window_days']) + ',' + str(par['metrics']['baseline_days']) + ',' + 
			str(par['metrics']['score']) + ',' + str(par['metrics']['hellinger']) + ',' + str(par['metrics']['time']))

		return par

	def CleanupParameters(self, par):
		"""
		saving space before pickling the model for later use
		"""
		self.logger.info('Performing parameter cleanup...')

		try:
			par['kdt'] = []
			par['knn'] = []

			par['m_ghost'] = []
			par['z_ghost'] = []
		except KeyError as e:
			pass

		return par

	def StaticTopicsGibbsSampling(self, par):
		"""
		performing gibbs sampling for static topics
		"""
		if par['T_static'] == 0:
			self.logger.info('The number of static topics is 0. Skipping static Gibbs sampling...')
			return par

		self.logger.info('Starting with Gibbs sampling for static documents')
		
		for iteration in range(par['max_static_iter']):
			for d in par['indices_static_doc']:
				for i in range(par['N'][d]):
					word_di = par['w'][d][i]
					old_topic = par['z'][d][i]
					assert par['m'][d][old_topic] >= 1
					assert par['n'][old_topic][word_di] >= 1
					assert par['n_sum'][old_topic] >= 1
					par['m'][d][old_topic] -= 1
					par['n'][old_topic][word_di] -= 1
					par['n_sum'][old_topic] -= 1

					topic_probabilities = []
					for topic_di in range(par['T_static']):
						topic_probability = 1.0 * (par['m'][d][topic_di] + par['alpha'][topic_di])
						topic_probability *= (par['n'][topic_di][word_di] + par['beta'][word_di])
						topic_probability /= (par['n_sum'][topic_di] + par['beta_sum'])
						topic_probabilities.append(topic_probability)
					sum_topic_probabilities = sum(topic_probabilities)
					assert sum_topic_probabilities > 0
					topic_probabilities = [p/sum_topic_probabilities for p in topic_probabilities]
					new_topic = list(np.random.multinomial(1, topic_probabilities, size=1)[0]).index(1)
					par['z'][d][i] = new_topic
					par['m'][d][new_topic] += 1
					par['n'][new_topic][word_di] += 1
					par['n_sum'][new_topic] += 1

				if d%1000 == 0:
					self.logger.info('  |-Done with iteration {iteration} and document {document}...'.format(iteration=iteration, document=d))
			self.logger.info('Done with iteration {iteration}'.format(iteration=iteration))

		self.logger.info('Done with Gibbs sampling for static documents')
		return par

	def EmergingTopicsGibbsSampling(self, par):
		"""
		performing gibbs sampling for emerging topics
		"""
		self.logger.info('Starting with Gibbs sampling for emerging documents')

		for iteration in range(par['max_emerging_iter']):
			for d in par['indices_emerging_doc']:
				if par['spatial_compactness_indicator_active'] and not par['doc_spatial_compactness_indicator'][d]:
					continue
				for i in range(par['N'][d]):
					word_di = par['w'][d][i]
					old_topic = par['z'][d][i]
					assert par['m'][d][old_topic] >= 1
					par['m'][d][old_topic] -= 1
					if old_topic in range(par['T_static'], par['T']) or par['options'].static_topics_evolve:
						assert par['n'][old_topic][word_di] >= 1
						assert par['n_sum'][old_topic] >= 1
						par['n'][old_topic][word_di] -= 1
						par['n_sum'][old_topic] -= 1

					# actual calculation of topic probabilities and topic sampling
					topic_probabilities = []
					for topic_di in range(par['T']):
						topic_probability = 1.0 * (par['m'][d][topic_di] + par['alpha'][topic_di])
						topic_probability *= (par['n'][topic_di][word_di] + par['beta'][word_di])
						topic_probability /= (par['n_sum'][topic_di] + par['beta_sum'])
						topic_probabilities.append(topic_probability)
					sum_topic_probabilities = sum(topic_probabilities)
					assert sum_topic_probabilities > 0
					topic_probabilities = [p/sum_topic_probabilities for p in topic_probabilities]
					new_topic = list(np.random.multinomial(1, topic_probabilities, size=1)[0]).index(1)
					par['z'][d][i] = new_topic
					par['m'][d][new_topic] += 1
					if new_topic in range(par['T_static'], par['T']) or par['options'].static_topics_evolve:
						par['n'][new_topic][word_di] += 1
						par['n_sum'][new_topic] += 1

					if par['spatial_compactness_indicator_active']:
						old_topic = par['z_ghost'][d][i]
						assert par['m_ghost'][d][old_topic] >= 1
						par['m_ghost'][d][old_topic] -= 1
						topic_probabilities = []
						for topic_di in range(par['T_static']):
							topic_probability = 1.0 * (par['m_ghost'][d][topic_di] + par['alpha'][topic_di])
							topic_probability *= (par['n'][topic_di][word_di] + par['beta'][word_di])
							topic_probability /= (par['n_sum'][topic_di] + par['beta_sum'])
							topic_probabilities.append(topic_probability)
						sum_topic_probabilities = sum(topic_probabilities)
						assert sum_topic_probabilities > 0
						topic_probabilities = [p/sum_topic_probabilities for p in topic_probabilities]
						new_topic = list(np.random.multinomial(1, topic_probabilities, size=1)[0]).index(1)
						par['z_ghost'][d][i] = new_topic
						par['m_ghost'][d][new_topic] += 1

				if d%1000 == 0:
					self.logger.info('  |-Done with iteration {iteration} and document {document} (location={lat},{lng})...'.format(iteration=iteration, document=d, lat=par['l'][d][0], lng=par['l'][d][1]))
			self.logger.info('Done with iteration {iteration}'.format(iteration=iteration))
			# if par['spatial_compactness_indicator_active'] and iteration < 5:
			#	self.CalculateSpatialCompactnessProbability(par)
			if par['spatial_compactness_indicator_active'] and iteration < 5:
				self.InferSpatialCompactnessUsingGFSS(par)
		
		self.logger.info('Done with Gibbs sampling for emerging documents')
		return par

	def VisualizeTopics(self, phi, words, num_topics, num_background_topics=25, num_foreground_topics=25, num_words_to_display=50, topics_filename='topics.pdf'):
		for t in range(num_topics):
			if sum(phi[t]) == 0:
				phi[t] = np.asarray([1.0/len(phi[t]) for _ in range(len(phi[t]))])
			else:
				phi[t] = np.asarray(phi[t])
				phi[t] = 1.0 * phi[t] / sum(phi[t])
		phi = np.asarray(phi)

		phi_viz = np.transpose(phi)

		###
		# viz_threshold = -np.sort(-np.amax(phi_viz, axis=1))[num_words_to_display-1]
		# decision_to_display = ~np.all(phi_viz < viz_threshold, axis=1)

		# phi_to_display = phi_viz[decision_to_display][:num_words_to_display,:]
		# words_to_display = [words[i] for i in range(len(decision_to_display)) if decision_to_display[i]][:num_words_to_display]

		# indices_to_display = [i[0] for i in sorted(enumerate(words_to_display), key=lambda x:x[1])]
		# phi_to_display = phi_to_display[indices_to_display]
		# words_to_display = sorted(words_to_display)
		###

		# background topics
		viz_threshold_b = -np.sort(-np.amax(phi_viz[:,:num_background_topics], axis=1))[num_words_to_display/2-1]
		decision_to_display_b = ~np.all(phi_viz[:,:num_background_topics] < viz_threshold_b, axis=1)

		phi_to_display_b = phi_viz[decision_to_display_b][:num_words_to_display/2-1,:]
		words_to_display_b = [words[i] for i in range(len(decision_to_display_b)) if decision_to_display_b[i]][:num_words_to_display/2-1]

		# foreground topics
		viz_threshold_f = -np.sort(-np.amax(phi_viz[:,num_background_topics:], axis=1))[num_words_to_display/2-1]
		decision_to_display_f = ~np.all(phi_viz[:,num_background_topics:] < viz_threshold_f, axis=1)
		decision_to_display_f = np.greater(decision_to_display_f, decision_to_display_b)

		phi_to_display_f = phi_viz[decision_to_display_f][:num_words_to_display/2,:]
		words_to_display_f = [words[i] for i in range(len(decision_to_display_f)) if decision_to_display_f[i]][:num_words_to_display/2]

		# join topics
		phi_to_display = np.concatenate((phi_to_display_b, phi_to_display_f))
		words_to_display = words_to_display_b + words_to_display_f

		# plot
		fig, ax = plt.subplots()
		heatmap = plt.pcolor(phi_to_display, cmap=plt.cm.Blues, alpha=0.8)
		plt.colorbar()

		#fig.set_size_inches(8, 11)
		ax.grid(False)
		ax.set_frame_on(True)

		ax.set_xticks(np.arange(phi_to_display.shape[1]) + 0.5, minor=False)
		ax.set_yticks(np.arange(phi_to_display.shape[0]) + 0.5, minor=False)
		ax.invert_yaxis()
		ax.xaxis.tick_top()
		#plt.xticks(rotation=45)
		
		for t in ax.xaxis.get_major_ticks():
		    t.tick1On = False
		    t.tick2On = False
		for t in ax.yaxis.get_major_ticks():
		    t.tick1On = False
		    t.tick2On = False

		column_labels = words_to_display
		row_labels = ['T' + str(i) for i in range(1,num_topics+1)]
		ax.set_xticklabels(row_labels, minor=False, fontsize=0)
		ax.set_yticklabels(column_labels, minor=False, fontsize=12)

		# plt.show()
		pp = PdfPages(topics_filename)
		pp.savefig()
		pp.close()

	def VisualizeSpatialEmergenceInSCSS(self, locations, posterior_probability, xlims=None, ylims=None, map_filename='map.pdf'):
		zipped_iter = zip(locations, posterior_probability)

		zipped_list = list(filter(lambda x: x[1] != 0.0, list(zipped_iter)))
		locations = [x[0] for x in zipped_list]
		posterior_probability = [x[1] for x in zipped_list]

		plt.figure()
		plt.scatter([l[1] for l in locations], [l[0] for l in locations], c='red', s=[450**x+50 for x in posterior_probability], alpha=0.5)
		
		plt.title('Posterior Probability Map of Emerging Topic')
		plt.xlabel('Longitude')
		plt.ylabel('Latitude')
		
		if xlims is not None:
			plt.xlim(xlims)
		if ylims is not None:
			plt.ylim(ylims)
		
		# plt.show()
		pp = PdfPages(map_filename)
		pp.savefig()
		pp.close()
