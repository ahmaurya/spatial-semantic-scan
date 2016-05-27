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
import random
import scipy.special
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pprint, pickle
from matplotlib.backends.backend_pdf import PdfPages
from ss import SemanticScan

def main_pnas():
	ss_pickle_path = '../results/pnas_tss/pnas_tss.pickle'
	ss_topics_path = '../results/plots_tss/topics_pnas.pdf'

	ss_pickle = open(ss_pickle_path, 'rb')
	par = pickle.load(ss_pickle)
	ss_pickle.close()

	ss = SpatiallyCompactSemanticScan()
	ss.VisualizeTopics(par['n'], par['word_token'], par['T'], 50, ss_topics_path)

def main_simulation():
	ss_pickle_path = '../results/simulation_tss/i40-pnas-simulation-tss.pickle'
	ss_pickle_path = '../../data/simulation/pnas_simulation.pickle'
	ss_topics_path = '../results/plots_tss/topics_simulation.pdf'

	ss_pickle = open(ss_pickle_path, 'rb')
	par = pickle.load(ss_pickle)
	ss_pickle.close()

	ss = SpatiallyCompactSemanticScan()
	ss.VisualizeTopics(par['n'], par['word_token'], par['T'], 50, ss_topics_path)

def main_twitter():
	ss_pickle_path = '../results/twitter_tss/twitter_tss.pickle'
	ss_topics_path = '../results/plots_tss/topics_twitter.pdf'

	ss_pickle = open(ss_pickle_path, 'rb')
	par = pickle.load(ss_pickle)
	ss_pickle.close()

	ss = SpatiallyCompactSemanticScan()
	ss.VisualizeTopics(par['n'], par['word_token'], par['T'], 50, ss_topics_path)

def main_yelp():
	ss_pickle_path = '../results/yelp_ss/yelp_ss.pickle'
	ss_topics_path = '../results/plots_ss/topics_yelp.pdf'
	ss_maps_path = '../results/plots_ss/map_yelp.pdf'
	
	ss_pickle = open(ss_pickle_path, 'rb')
	par = pickle.load(ss_pickle)
	ss_pickle.close()

	ss = SemanticScan()
	for i in range(15,25):
		ss.VisualizeTopics(par['n'], par['word_token'], par['T'], 25, 25, i, ss_topics_path.replace('yelp', 'yelp' + str(i)))
	#ss.VisualizeSpatialEmergenceInSCSS(par['l'], par['posterior_probability_emerging_topic'], [-115.16,-115.15], [36.145,36.16], ss_maps_path)

def main_edc():
	ss_pickle_path = '../results/edc_ss/edc_ss.pickle'
	ss_topics_path = '../results/plots_ss/topics_edc.pdf'
	ss_maps_path = '../results/plots_ss/map_edc.pdf'

	ss_pickle = open(ss_pickle_path, 'rb')
	par = pickle.load(ss_pickle)
	ss_pickle.close()

	ss = SemanticScan()
	ss.VisualizeTopics(par['n'], par['word_token'], par['T'], 25, 25, 25, ss_topics_path)
	#ss.VisualizeSpatialEmergenceInSCSS(par['l'], par['posterior_probability_emerging_topic'], [-110,-50], [36,45], ss_maps_path)

if __name__ == "__main__":
	# main_pnas()
	# main_simulation()
	# main_twitter()
	main_yelp()
	main_edc()
