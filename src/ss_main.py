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


import pickle
from optparse import OptionParser, Option
from ss import SemanticScan
import time

def main_edc_ss_moving_window():
	parser = OptionParser()
	parser.add_option("-a", "--ifspatialscan", dest="ifspatialscan", action="store_true", default=True, help="Whether execute spatial scan after topic modeling")
	parser.add_option("-s", "--ifspatialcompactness", dest="ifspatialcompactness", action="store_true", default=False, help="Whether execute Semantic Scan or Spatially Compact Semantic Scan")
	parser.add_option("-c", "--normalize_timestamps", dest="normalize_timestamps", action="store_false", default=False, help="Whether to normalize timestamps")
	parser.add_option("-k", "--k", dest="k", type="int", default=30, help="K - Neighborhood size in GFSS")
	parser.add_option("-q", "--sparsity", dest="sparsity", type="float", default=0.5, help="Sparsity")
	parser.add_option("-u", "--startstatic", dest="startstatic", type="float", default=-1.0, help="Static Topic Start Timestamp")
	parser.add_option("-v", "--endstatic", dest="endstatic", type="float", default=-1.0, help="Static Topic End Timestamp")
	parser.add_option("-w", "--startemerging", dest="startemerging", type="float", default=-1.0, help="Emerging Topic Start Timestamp")
	parser.add_option("-x", "--endemerging", dest="endemerging", type="float", default=-1.0, help="Emerging Topic End Timestamp")
	parser.add_option("-m", "--nstatictopics", dest="nstatictopics", type="int", default=25, help="Number of Static Topics")
	parser.add_option("-n", "--nemergingtopics", dest="nemergingtopics", type="int", default=25, help="Number of Emerging Topics")
	parser.add_option("-i", "--staticiterations", dest="staticiterations", type="int", default=10, help="Number of Static Iterations")
	parser.add_option("-j", "--emergingiterations", dest="emergingiterations", type="int", default=10, help="Number of Emerging Iterations")
	parser.add_option("-l", "--onlineiterations", dest="onlineiterations", type="int", default=10, help="Number of Iterations in Online Document Assignment")
	parser.add_option("-d", "--datapath", dest="datapath", type="string", default='../../data/edc/simulation/', help="Data Path")
	parser.add_option("-r", "--resultspath", dest="resultspath", type="string", default='../results/edc_ss/', help="Results Path")
	parser.add_option("-b", "--backgrndfilename", dest="backgrndfilename", type="string", default='simulation-background-786.csv', help="Background Corpus Filename")
	parser.add_option("-f", "--foregrndfilename", dest="foregrndfilename", type="string", default='simulation-foreground-786-0.csv', help="Foreground Corpus Filename")
	parser.add_option("-z", "--zipcodesfilename", dest="zipcodesfilename", type="string", default='../zipcode.csv', help="Zipcodes Filename")
	parser.add_option("-p", "--picklefilename", dest="picklefilename", type="string", default='edc_ss.pickle', help="Pickle Filename")
	parser.add_option("-g", "--windowdays", dest="windowdays", type="int", default=3, help="Window days")
	parser.add_option("-e", "--baselinedays", dest="baselinedays", type="int", default=30, help="Baseline days")
	parser.add_option("-o", "--static_topics_evolve", dest="static_topics_evolve", action="store_true", default=False, help="Whether static topics can evolve in the emeging topic detection phase")
	parser.add_option("-t", "--scan_type", dest="scan_type", type="int", default=1, help="Type of scan: 0=spatially constrained LTSS, 1=circular spatial scan, 2=naive count-based scan")
	(options, args) = parser.parse_args()

	background_path = options.datapath + options.backgrndfilename
	foreground_path = options.datapath + options.foregrndfilename
	zipcodes_path = options.datapath + options.zipcodesfilename
	pickle_path = options.resultspath + options.picklefilename

	ss = SemanticScan()
	documents = ss.GetEDChiefComplaintCorpus(background_path, foreground_path, zipcodes_path)
	par = ss.InitializeStaticParameters(documents, options)
	par = ss.StaticTopicsGibbsSampling(par)
	for i in range(30, 360):
		par = ss.ReinitializeEmergingParameters(par, start_day=i)
		par['metrics']['time'] = int(round(time.time() * 1000))
		par = ss.EmergingTopicsGibbsSampling(par)
		par['metrics']['time'] = int(round(time.time() * 1000)) - par['metrics']['time']
		par = ss.CalculateDocumentAssignment(par)
		par = ss.PerformSpatialScan(par)
		par = ss.GetMetrics(par)
	par['theta'], par['phi'] = ss.ComputePosteriorEstimatesOfThetaAndPhi(par)
	par = ss.CleanupParameters(par)

def main_yelp_ss_moving_window():
	parser = OptionParser()
	parser.add_option("-a", "--ifspatialscan", dest="ifspatialscan", action="store_true", default=True, help="Whether execute spatial scan after topic modeling")
	parser.add_option("-s", "--ifspatialcompactness", dest="ifspatialcompactness", action="store_true", default=False, help="Whether execute Semantic Scan or Spatially Compact Semantic Scan")
	parser.add_option("-c", "--normalize_timestamps", dest="normalize_timestamps", action="store_false", default=False, help="Whether to normalize timestamps")
	parser.add_option("-k", "--k", dest="k", type="int", default=30, help="K - Neighborhood size in GFSS")
	parser.add_option("-q", "--sparsity", dest="sparsity", type="float", default=0.5, help="Sparsity")
	parser.add_option("-u", "--startstatic", dest="startstatic", type="float", default=-1.0, help="Static Topic Start Timestamp")
	parser.add_option("-v", "--endstatic", dest="endstatic", type="float", default=-1.0, help="Static Topic End Timestamp")
	parser.add_option("-w", "--startemerging", dest="startemerging", type="float", default=-1.0, help="Emerging Topic Start Timestamp")
	parser.add_option("-x", "--endemerging", dest="endemerging", type="float", default=-1.0, help="Emerging Topic End Timestamp")
	parser.add_option("-m", "--nstatictopics", dest="nstatictopics", type="int", default=25, help="Number of Static Topics")
	parser.add_option("-n", "--nemergingtopics", dest="nemergingtopics", type="int", default=25, help="Number of Emerging Topics")
	parser.add_option("-i", "--staticiterations", dest="staticiterations", type="int", default=10, help="Number of Static Iterations")
	parser.add_option("-j", "--emergingiterations", dest="emergingiterations", type="int", default=10, help="Number of Emerging Iterations")
	parser.add_option("-l", "--onlineiterations", dest="onlineiterations", type="int", default=10, help="Number of Iterations in Online Document Assignment")
	parser.add_option("-d", "--datapath", dest="datapath", type="string", default='../../data/yelp/simulation/', help="Data Path")
	parser.add_option("-r", "--resultspath", dest="resultspath", type="string", default='../results/yelp_ss/', help="Results Path")
	parser.add_option("-b", "--backgrndfilename", dest="backgrndfilename", type="string", default='simulation-background-indian.csv', help="Background Corpus Filename")
	parser.add_option("-f", "--foregrndfilename", dest="foregrndfilename", type="string", default='simulation-foreground-indian.csv', help="Foreground Corpus Filename")
	parser.add_option("-z", "--zipcodesfilename", dest="zipcodesfilename", type="string", default='../zipcode.csv', help="Zipcodes Filename")
	parser.add_option("-p", "--picklefilename", dest="picklefilename", type="string", default='yelp_ss.pickle', help="Pickle Filename")
	parser.add_option("-g", "--windowdays", dest="windowdays", type="int", default=3, help="Window days")
	parser.add_option("-e", "--baselinedays", dest="baselinedays", type="int", default=30, help="Baseline days")
	parser.add_option("-o", "--static_topics_evolve", dest="static_topics_evolve", action="store_true", default=False, help="Whether static topics can evolve in the emeging topic detection phase")
	parser.add_option("-t", "--scan_type", dest="scan_type", type="int", default=1, help="Type of scan: 0=spatially constrained LTSS, 1=circular spatial scan, 2=naive count-based scan")
	(options, args) = parser.parse_args()

	background_path = options.datapath + options.backgrndfilename
	foreground_path = options.datapath + options.foregrndfilename
	zipcodes_path = options.datapath + options.zipcodesfilename
	pickle_path = options.resultspath + options.picklefilename

	ss = SemanticScan()
	documents = ss.GetEDChiefComplaintCorpus(background_path, foreground_path, zipcodes_path)
	par = ss.InitializeStaticParameters(documents, options)
	par = ss.StaticTopicsGibbsSampling(par)
	for i in range(30, 350):
		par = ss.ReinitializeEmergingParameters(par, start_day=i)
		par['metrics']['time'] = int(round(time.time() * 1000))
		par = ss.EmergingTopicsGibbsSampling(par)
		par['metrics']['time'] = int(round(time.time() * 1000)) - par['metrics']['time']
		par = ss.CalculateDocumentAssignment(par)
		par = ss.PerformSpatialScan(par)
		par = ss.GetMetrics(par)
	par['theta'], par['phi'] = ss.ComputePosteriorEstimatesOfThetaAndPhi(par)
	par = ss.CleanupParameters(par)

def main_edc():
	parser = OptionParser()
	parser.add_option("-a", "--ifspatialcompactness", dest="ifspatialcompactness", action="store_true", default=False, help="Whether execute Semantic Scan or Spatially Compact Semantic Scan")
	parser.add_option("-s", "--ifspatialscan", dest="ifspatialscan", action="store_true", default=True, help="Whether execute spatial scan after topic modeling")
	parser.add_option("-c", "--normalize_timestamps", dest="normalize_timestamps", action="store_false", default=False, help="Whether to normalize timestamps")
	parser.add_option("-k", "--k", dest="k", type="int", default=100, help="K - Neighborhood size in GFSS")
	parser.add_option("-u", "--startstatic", dest="startstatic", type="float", default=-1.0, help="Static Topic Start Timestamp")
	parser.add_option("-v", "--endstatic", dest="endstatic", type="float", default=-1.0, help="Static Topic End Timestamp")
	parser.add_option("-w", "--startemerging", dest="startemerging", type="float", default=-1.0, help="Emerging Topic Start Timestamp")
	parser.add_option("-x", "--endemerging", dest="endemerging", type="float", default=-1.0, help="Emerging Topic End Timestamp")
	parser.add_option("-m", "--nstatictopics", dest="nstatictopics", type="int", default=25, help="Number of Static Topics")
	parser.add_option("-n", "--nemergingtopics", dest="nemergingtopics", type="int", default=25, help="Number of Emerging Topics")
	parser.add_option("-i", "--staticiterations", dest="staticiterations", type="int", default=10, help="Number of Static Iterations")
	parser.add_option("-j", "--emergingiterations", dest="emergingiterations", type="int", default=10, help="Number of Emerging Iterations")
	parser.add_option("-l", "--onlineiterations", dest="onlineiterations", type="int", default=10, help="Number of Iterations in Online Document Assignment")
	parser.add_option("-d", "--datapath", dest="datapath", type="string", default='../../data/edc/simulation/', help="Data Path")
	parser.add_option("-r", "--resultspath", dest="resultspath", type="string", default='../results/edc_ll/', help="Results Path")
	parser.add_option("-b", "--backgrndfilename", dest="backgrndfilename", type="string", default='simulation-background-623.csv', help="Background Corpus Filename")
	parser.add_option("-f", "--foregrndfilename", dest="foregrndfilename", type="string", default='simulation-foreground-623-0.csv', help="Foreground Corpus Filename")
	parser.add_option("-z", "--zipcodesfilename", dest="zipcodesfilename", type="string", default='../zipcode.csv', help="Zipcodes Filename")
	parser.add_option("-p", "--picklefilename", dest="picklefilename", type="string", default='edc_ll.pickle', help="Pickle Filename")
	parser.add_option("-g", "--windowdays", dest="windowdays", type="int", default=3, help="Window days")
	parser.add_option("-e", "--baselinedays", dest="baselinedays", type="int", default=30, help="Baseline days")
	parser.add_option("-o", "--static_topics_evolve", dest="static_topics_evolve", action="store_true", default=True, help="Whether static topics can evolve in the emeging topic detection phase")
	parser.add_option("-t", "--scan_type", dest="scan_type", type="int", default=1, help="Type of scan: 0=spatially constrained LTSS, 1=circular spatial scan, 2=naive count-based scan")
	(options, args) = parser.parse_args()

	background_path = options.datapath + options.backgrndfilename
	foreground_path = options.datapath + options.foregrndfilename
	zipcodes_path = options.datapath + options.zipcodesfilename
	pickle_path = options.resultspath + options.picklefilename

	ss = SemanticScan()
	documents = ss.GetEDChiefComplaintCorpus(background_path, foreground_path, zipcodes_path)
	par = ss.InitializeStaticParameters(documents, options)
	par = ss.StaticTopicsGibbsSampling(par)
	par = ss.InitializeEmergingParameters(par)
	par = ss.EmergingTopicsGibbsSampling(par)
	#par = ss.PerformSpatialScan(par)
	par['theta'], par['phi'] = ss.ComputePosteriorEstimatesOfThetaAndPhi(par)
	#par = ss.CleanupParameters(par)
	ss_pickle = open(pickle_path, 'wb')
	pickle.dump(par, ss_pickle)
	ss_pickle.close()

def main_yelp():
	parser = OptionParser()
	parser.add_option("-a", "--ifspatialcompactness", dest="ifspatialcompactness", action="store_true", default=True, help="Whether execute Semantic Scan or Spatially Compact Semantic Scan")
	parser.add_option("-s", "--ifspatialscan", dest="ifspatialscan", action="store_true", default=False, help="Whether execute spatial scan after topic modeling")
	parser.add_option("-c", "--normalize_timestamps", dest="normalize_timestamps", action="store_false", default=False, help="Whether to normalize timestamps")
	parser.add_option("-k", "--k", dest="k", type="int", default=1000, help="K - Neighborhood size in GFSS")
	parser.add_option("-u", "--startstatic", dest="startstatic", type="float", default=-1.0, help="Static Topic Start Timestamp")
	parser.add_option("-v", "--endstatic", dest="endstatic", type="float", default=-1.0, help="Static Topic End Timestamp")
	parser.add_option("-w", "--startemerging", dest="startemerging", type="float", default=-1.0, help="Emerging Topic Start Timestamp")
	parser.add_option("-x", "--endemerging", dest="endemerging", type="float", default=-1.0, help="Emerging Topic End Timestamp")
	parser.add_option("-m", "--nstatictopics", dest="nstatictopics", type="int", default=25, help="Number of Static Topics")
	parser.add_option("-n", "--nemergingtopics", dest="nemergingtopics", type="int", default=25, help="Number of Emerging Topics")
	parser.add_option("-i", "--staticiterations", dest="staticiterations", type="int", default=30, help="Number of Static Iterations")
	parser.add_option("-j", "--emergingiterations", dest="emergingiterations", type="int", default=30, help="Number of Emerging Iterations")
	parser.add_option("-l", "--onlineiterations", dest="onlineiterations", type="int", default=10, help="Number of Iterations in Online Document Assignment")
	parser.add_option("-d", "--datapath", dest="datapath", type="string", default='../../data/yelp/', help="Data Path")
	parser.add_option("-r", "--resultspath", dest="resultspath", type="string", default='../results/yelp_tss/', help="Results Path")
	parser.add_option("-f", "--corpusfilename", dest="corpusfilename", type="string", default='yelp_academic_dataset_review.json', help="Corpus Filename")
	parser.add_option("-b", "--businessfilename", dest="businessfilename", type="string", default='yelp_academic_dataset_business.json', help="Business Filename")
	parser.add_option("-z", "--stopwordsfilename", dest="stopwordsfilename", type="string", default='yelp_stopwords.txt', help="Stopwords Filename")
	parser.add_option("-p", "--picklefilename", dest="picklefilename", type="string", default='yelp_tss.pickle', help="Pickle Filename")
	parser.add_option("-g", "--windowdays", dest="windowdays", type="int", default=3, help="Window days")
	parser.add_option("-e", "--baselinedays", dest="baselinedays", type="int", default=30, help="Baseline days")
	(options, args) = parser.parse_args()

	corpus_path = options.datapath + options.corpusfilename
	business_path = options.datapath + options.businessfilename
	stopwords_path = options.datapath + options.stopwordsfilename
	pickle_path = options.resultspath + options.picklefilename

	ss = SemanticScan()
	documents = ss.GetYelpCorpus(corpus_path, business_path, stopwords_path)
	par = ss.InitializeStaticParameters(documents, options)
	par = ss.StaticTopicsGibbsSampling(par)
	par = ss.InitializeEmergingParameters(par)
	par = ss.EmergingTopicsGibbsSampling(par)
	par['theta'], par['phi'] = ss.ComputePosteriorEstimatesOfThetaAndPhi(par)
	par = ss.CleanupParameters(par)
	ss_pickle = open(pickle_path, 'wb')
	pickle.dump(par, ss_pickle)
	ss_pickle.close()

def main_pnas():
	parser = OptionParser()
	parser.add_option("-a", "--ifspatialcompactness", dest="ifspatialcompactness", action="store_true", default=True, help="Whether execute Semantic Scan or Spatially Compact Semantic Scan")
	parser.add_option("-s", "--ifspatialscan", dest="ifspatialscan", action="store_true", default=False, help="Whether execute spatial scan after topic modeling")
	parser.add_option("-c", "--normalize_timestamps", dest="normalize_timestamps", action="store_false", default=False, help="Whether to normalize timestamps")
	parser.add_option("-k", "--k", dest="k", type="int", default=100, help="K - Neighborhood size in GFSS")
	parser.add_option("-u", "--startstatic", dest="startstatic", type="float", default=0.0, help="Static Topic Start Timestamp")
	parser.add_option("-v", "--endstatic", dest="endstatic", type="float", default=0.5, help="Static Topic End Timestamp")
	parser.add_option("-w", "--startemerging", dest="startemerging", type="float", default=0.5, help="Emerging Topic Start Timestamp")
	parser.add_option("-x", "--endemerging", dest="endemerging", type="float", default=1.0, help="Emerging Topic End Timestamp")
	parser.add_option("-m", "--nstatictopics", dest="nstatictopics", type="int", default=9, help="Number of Static Topics")
	parser.add_option("-n", "--nemergingtopics", dest="nemergingtopics", type="int", default=1, help="Number of Emerging Topics")
	parser.add_option("-i", "--staticiterations", dest="staticiterations", type="int", default=10, help="Number of Static Iterations")
	parser.add_option("-j", "--emergingiterations", dest="emergingiterations", type="int", default=10, help="Number of Emerging Iterations")
	parser.add_option("-l", "--onlineiterations", dest="onlineiterations", type="int", default=10, help="Number of Iterations in Online Document Assignment")
	parser.add_option("-d", "--datapath", dest="datapath", type="string", default='../../data/pnas/', help="Data Path")
	parser.add_option("-r", "--resultspath", dest="resultspath", type="string", default='../results/pnas_tss/', help="Results Path")
	parser.add_option("-f", "--corpusfilename", dest="corpusfilename", type="string", default='alltitles', help="Corpus Filename")
	parser.add_option("-t", "--timestampsfilename", dest="timestampsfilename", type="string", default='alltimes', help="Timestamps Filename")
	parser.add_option("-q", "--locationsfilename", dest="locationsfilename", type="string", default='alllocations', help="Location Filename")
	parser.add_option("-z", "--stopwordsfilename", dest="stopwordsfilename", type="string", default='allstopwords', help="Stopwords Filename")
	parser.add_option("-p", "--picklefilename", dest="picklefilename", type="string", default='pnas_tss.pickle', help="Pickle Filename")
	parser.add_option("-g", "--windowdays", dest="windowdays", type="int", default=3, help="Window days")
	parser.add_option("-e", "--baselinedays", dest="baselinedays", type="int", default=30, help="Baseline days")
	(options, args) = parser.parse_args()

	corpus_path = options.datapath + options.corpusfilename
	timestamps_path = options.datapath + options.timestampsfilename
	locations_path = options.datapath + options.locationsfilename
	stopwords_path = options.datapath + options.stopwordsfilename
	pickle_path = options.resultspath + options.picklefilename

	ss = SemanticScan()
	documents = ss.GetPnasCorpus(corpus_path, timestamps_path, stopwords_path, locations_path)
	par = ss.InitializeStaticParameters(documents, options)
	par = ss.StaticTopicsGibbsSampling(par)
	par = ss.InitializeEmergingParameters(par)
	par = ss.EmergingTopicsGibbsSampling(par)
	par['theta'], par['phi'] = ss.ComputePosteriorEstimatesOfThetaAndPhi(par)
	par = ss.CleanupParameters(par)
	ss_pickle = open(pickle_path, 'wb')
	pickle.dump(par, ss_pickle)
	ss_pickle.close()

def main_simulation():
	parser = OptionParser()
	parser.add_option("-a", "--ifspatialcompactness", dest="ifspatialcompactness", action="store_true", default=True, help="Whether execute Semantic Scan or Spatially Compact Semantic Scan")
	parser.add_option("-s", "--ifspatialscan", dest="ifspatialscan", action="store_true", default=False, help="Whether execute spatial scan after topic modeling")
	parser.add_option("-c", "--normalize_timestamps", dest="normalize_timestamps", action="store_false", default=False, help="Whether to normalize timestamps")
	parser.add_option("-k", "--k", dest="k", type="int", default=100, help="K - Neighborhood size in GFSS")
	parser.add_option("-u", "--startstatic", dest="startstatic", type="float", default=0.0, help="Static Topic Start Timestamp")
	parser.add_option("-v", "--endstatic", dest="endstatic", type="float", default=0.5, help="Static Topic End Timestamp")
	parser.add_option("-w", "--startemerging", dest="startemerging", type="float", default=0.5, help="Emerging Topic Start Timestamp")
	parser.add_option("-x", "--endemerging", dest="endemerging", type="float", default=1.0, help="Emerging Topic End Timestamp")
	parser.add_option("-m", "--nstatictopics", dest="nstatictopics", type="int", default=19, help="Number of Static Topics")
	parser.add_option("-n", "--nemergingtopics", dest="nemergingtopics", type="int", default=1, help="Number of Emerging Topics")
	parser.add_option("-i", "--staticiterations", dest="staticiterations", type=int, default=30, help="Max Iterations of Static Gibbs Sampling")
	parser.add_option("-j", "--emergingiterations", dest="emergingiterations", type=int, default=30, help="Max Iterations of Emerging Gibbs Sampling")
	parser.add_option("-l", "--onlineiterations", dest="onlineiterations", type="int", default=10, help="Number of Iterations in Online Document Assignment")
	parser.add_option("-d", "--datapath", dest="datapath", type="string", default='../../data/simulation/', help="Data Path")
	parser.add_option("-r", "--resultspath", dest="resultspath", type="string", default='../results/simulation_tss/', help="Results Path")
	parser.add_option("-f", "--corpusfilename", dest="corpusfilename", type="string", default='simulation-titles.txt', help="Corpus Filename")
	parser.add_option("-t", "--timestampsfilename", dest="timestampsfilename", type="string", default='simulation-timestamps.txt', help="Timestamps Filename")
	parser.add_option("-q", "--locationsfilename", dest="locationsfilename", type="string", default='simulation-locations.txt', help="Location Filename")
	parser.add_option("-y", "--dictionaryfilename", dest="dictionaryfilename", type="string", default='simulation-dictionary.txt', help="Dictionary Filename")
	parser.add_option("-b", "--estimatedaffecteddocs", dest="estimatedaffecteddocs", type="string", default='tss-affecteddocs.txt', help="Estimated Affected Documents Filename")
	parser.add_option("-z", "--stopwordsfilename", dest="stopwordsfilename", type="string", default='simulation-stopwords.txt', help="Stopwords Filename")
	parser.add_option("-p", "--picklefilename", dest="picklefilename", type="string", default='pnas_simulation_tss.pickle', help="Pickle Filename")
	parser.add_option("-g", "--windowdays", dest="windowdays", type="int", default=3, help="Window days")
	parser.add_option("-e", "--baselinedays", dest="baselinedays", type="int", default=30, help="Baseline days")
	(options, args) = parser.parse_args()

	data_path = options.datapath
	results_path = options.resultspath
	corpus_path = data_path + options.corpusfilename
	timestamps_path = data_path + options.timestampsfilename
	locations_path = data_path + options.locationsfilename
	stopwords_path = data_path + options.stopwordsfilename
	dictionary_path = data_path + options.dictionaryfilename
	pickle_path = results_path + options.picklefilename
	estimated_affected_docs_path = results_path + options.estimatedaffecteddocs

	ss = SemanticScan()
	documents = ss.GetPnasCorpus(corpus_path, timestamps_path, stopwords_path, locations_path)
	dictionary = str.split(open(dictionary_path, 'r').read())
	par = ss.InitializeStaticParameters(documents, options)
	par = ss.StaticTopicsGibbsSampling(par)
	par = ss.InitializeEmergingParameters(par)
	par = ss.EmergingTopicsGibbsSampling(par)
	par['theta'], par['phi'] = ss.ComputePosteriorEstimatesOfThetaAndPhi(par)
	par = ss.CleanupParameters(par)

	ss_pickle = open(pickle_path, 'wb')
	pickle.dump(par, ss_pickle)
	ss_pickle.close()
	
	estimated_affected_docs = open(estimated_affected_docs_path, 'w')
	estimated_affected_docs.write('\n'.join([str(d) for d in par['spatial_coherence_probability']]))
	estimated_affected_docs.close()

def main_twitter():
	parser = OptionParser()
	parser.add_option("-a", "--ifspatialcompactness", dest="ifspatialcompactness", action="store_true", default=True, help="Whether execute Semantic Scan or Spatially Compact Semantic Scan")
	parser.add_option("-s", "--ifspatialscan", dest="ifspatialscan", action="store_true", default=False, help="Whether execute spatial scan after topic modeling")
	parser.add_option("-c", "--normalize_timestamps", dest="normalize_timestamps", action="store_false", default=False, help="Whether to normalize timestamps")
	parser.add_option("-k", "--k", dest="k", type="int", default=100, help="K - Neighborhood size in GFSS")
	parser.add_option("-u", "--startstatic", dest="startstatic", type="float", default=0.0, help="Static Topic Start Timestamp")
	parser.add_option("-v", "--endstatic", dest="endstatic", type="float", default=0.5, help="Static Topic End Timestamp")
	parser.add_option("-w", "--startemerging", dest="startemerging", type="float", default=0.5, help="Emerging Topic Start Timestamp")
	parser.add_option("-x", "--endemerging", dest="endemerging", type="float", default=1.0, help="Emerging Topic End Timestamp")
	parser.add_option("-m", "--nstatictopics", dest="nstatictopics", type="int", default=9, help="Number of Static Topics")
	parser.add_option("-n", "--nemergingtopics", dest="nemergingtopics", type="int", default=1, help="Number of Emerging Topics")
	parser.add_option("-i", "--staticiterations", dest="staticiterations", type="int", default=10, help="Number of Static Iterations")
	parser.add_option("-j", "--emergingiterations", dest="emergingiterations", type="int", default=10, help="Number of Emerging Iterations")
	parser.add_option("-l", "--onlineiterations", dest="onlineiterations", type="int", default=10, help="Number of Iterations in Online Document Assignment")
	parser.add_option("-d", "--datapath", dest="datapath", type="string", default='../../data/twitter/', help="Data Path")
	parser.add_option("-r", "--resultspath", dest="resultspath", type="string", default='../results/twitter_tss/', help="Results Path")
	parser.add_option("-f", "--corpusfilename", dest="corpusfilename", type="string", default='twitter_2013_08_03.csv', help="Corpus Filename")
	parser.add_option("-z", "--stopwordsfilename", dest="stopwordsfilename", type="string", default='allstopwords', help="Stopwords Filename")
	parser.add_option("-p", "--picklefilename", dest="picklefilename", type="string", default='twitter_tss_2013_08_03.pickle', help="Pickle Filename")
	parser.add_option("-g", "--windowdays", dest="windowdays", type="int", default=3, help="Window days")
	parser.add_option("-e", "--baselinedays", dest="baselinedays", type="int", default=30, help="Baseline days")
	(options, args) = parser.parse_args()

	data_path = options.datapath
	results_path = options.resultspath
	corpus_path = data_path + options.corpusfilename
	timestamps_path = data_path + options.timestampsfilename
	locations_path = data_path + options.locationsfilename
	stopwords_path = data_path + options.stopwordsfilename
	pickle_path = results_path + options.picklefilename

	ss = SemanticScan()
	documents = ss.GetTwitterCorpus(corpus_path, stopwords_path)
	par = ss.InitializeStaticParameters(documents, options)
	par = ss.StaticTopicsGibbsSampling(par)
	par = ss.InitializeEmergingParameters(par)
	par = ss.EmergingTopicsGibbsSampling(par)
	par['theta'], par['phi'] = ss.ComputePosteriorEstimatesOfThetaAndPhi(par)
	par = ss.CleanupParameters(par)
	ss_pickle = open(pickle_path, 'wb')
	pickle.dump(par, ss_pickle)
	ss_pickle.close()

if __name__ == "__main__":
	#main_pnas()
	#main_simulation()
	#main_twitter()
	#main_yelp()
	main_edc()
	#main_edc_ss_moving_window()
	#main_yelp_ss_moving_window()
