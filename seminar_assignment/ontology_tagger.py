import parser
import re
from nltk import tokenize
from collections import Counter
import os
import nltk
import gensim
from gensim.models import Word2Vec

class Ontology:
	def __init__(self, model):
		self.ontology = {
				'science' : {

					'physics' : {
						'mechanics' : {
							'topic_words' : ['mechanics', 'mechanical', 'speed', 'distance', 'acceleration', 'instantaneous', 'force',
										     'newton', 'weight', 'inertia', 'momentum', 'impulse', 'energy', 'elastic collision',
										     'inelastic collision', 'power', 'kilogram', 'meter', 'lift', 'joule', 'kinetic energy',
										     'mass', 'time'],
							'talks'       : list()
						},
						'electricity' : {
							'topic_words' : ['electrostatic', 'electric', 'electricity', 'electromagnetism', 'electronic',
											 'ohm', 'amper', 'coulomb', 'charge', 'conduction', 'induction', 'conductors',
											 'volt', 'watt', 'current', 'resistance', 'resistivity', 'circuit', 'resistor',
											 'battery', 'ammeter', 'diode'],
							'talks'       : list()
						},
						'thermodynamics' : {
							'topic_words' : ['thermodinamic', 'temperature', 'heat', 'isobaric', 'isochoric', 'isothermal', 'adiabatic',
											 'celsius', 'farenheit', 'kelvin', 'entropy', 'carnot', 'calorie', 'conserve',
											 'joule', 'gas', 'piston', 'pressure', 'engine', 'sublimation', 'evaporation', 'condensation',
											 'freezing', 'melting'],
							'talks'       : list()
						},
						'optics'         : {
							'topic_words' : ['optic', 'reflection', 'refraction', 'fermat', 'lens', 'converging', 'diverging',
											 'light', 'convex', 'concavev', 'mirror', 'focal', 'inverted'],
							'talks'       : list()
						},
						'cosmology'      : {
							'topic_words' : ['cosmos', 'light-year', 'Big Bang', 'dark matter', 'dark energy', 'astronomical',
											 'universe', 'asteroids', 'comets', 'sun', 'constellation', 'galaxies', 'supernova',
											 'star', 'black hole'],
							'talks'       : list()
						}
					},

					'chemistry' : {
						'organic' : {
							'topic_words' : ['organic', 'polymer', 'alkali', 'allotropes', 'alkali', 'alkaline', 'alkanes', 'alkenes',
											 'anode', 'autocatalysis', 'avogadro', 'catalyst', 'cathode', 'concentration',
											 'condensation', 'acid', 'conjugate base', 'dissociation', 'distilled', 'electrodes',
											 'electrolysis', 'electrolyte', 'electronegativity', 'esterification', 'halogens',
											 'neutralisation', 'hydrocarbons'],
							'talks'        : list()
						},
						'inorganic' : {
							'topic_words' : ['inorganic', 'atomic number', 'atomic mass', 'molar mass', 'planck', 'electron',
											 'chalcogenes', 'halogenes', 'nobel gasses', 'intermolecular', 'acetone', 'benzene',
											 'dynamic equilibrium', 'ideal gas', 'amphoteric', 'oxidation', 'reduction',
											 'electrodes', 'faraday'],
							'talks'       : list()
						}
					},

					'biology' : {
						'botanyy' : {
							'topic_words' : ['botany', 'plant', 'seed', 'gimnosperms', 'angiosperms', 'monocots', 'dicots',
											 'fruit', 'flower', 'cuticle', 'pollen', 'stomata', 'cone', 'stamen', 'fertilization',
											 'cellulose', 'photosyntesis', 'cellular respiration', 'osmosis', 'rhizosphere',
											 'xylem', 'phototropism'],
							'talks'       : list()
						},
						'anathomy' : {
							'topic_words' : ['anathomy', 'assimilation', 'epidermis', 'nervous', 'dialysis', 'homeostasis',
											 'physiology', 'proteins', 'adipose tissue', 'plasma', 'thymine', 'gene',
											 'genetics', 'recessive', 'homozygous', 'genotype', 'phenotype', 'exon',
											 'mutation', 'mutagen', 'genome', 'plasmid', 'pedigree', 'sex chromosome',
											 'replication', 'DNA', 'RNA', 'digestion', 'respiratory', 'reproductive',
											 'excretory', 'vertebrate'],
							'talks'       : list()
						},
						'microbiology' : {
							'topic_words' : ['microbiology', 'prokaryote', 'halophiles', 'methanogenes', 'bacteria'
											 'pathogen', 'mutualism', 'antibiotics', 'disease', 'immunity', 'virus', 
											 'organism', 'eukarya', 'methanogenes', 'bacilli', 'cocci', 'infection',
											 'penicillin', 'alive', 'antigen'],
							'talks'       : list()
						}
					},

					'mathematics' : {
						'algebra' : {
							'topic_words' : ['algebra', 'absolute value', 'additive inverse', 'coefficient', 'constant',
											 'coordinate', 'coordinate plane', 'discriminant', 'exponent', 'independent system',
											 'irrational number', 'linear', 'monomial', 'parabola', 'perfect square',
											 'perpendicular', 'polynomial', 'quadratic', 'decimal', 'x-axis', 'y-axes',
											 'domain', 'radical', 'matrix'],
							'talks'       : list()
						},
						'geometry' : {
							'topic_words' : ['geometry', 'perpendicular', 'parallel', 'shape', 'angle', 'point', 'line',
											 'circle', 'triangle', 'square', 'area', 'volume', 'surface', 'diagonal',
											 'altitude', 'median', 'polygon', 'vertex', 'isosceles', 'equilateral', 
											 'degrees', 'obtuse'],
							'talks'        : list()
						}
					},
					'computer science' : {
						'artificial intelligence' : {
							'topic_words' : ['artificial intelligence', 'robot', 'robotics', 'AI', 'machine learning',
											 'neural network', 'reasoning', 'automate', 'back propagation', 'bigram',
											 'CNF', 'learning', 'decision', 'deep', 'finite automaton', 'forward chaining',
											 'grammar', 'heuristic', 'layer', 'neural', 'lemma', 'planning', 'perceptron',
											 'search tree', 'synapse', 'search'],
							'talks'       : list()
						},
						'networking' : {
							'topic_words' : ['bluetooth', 'networking', 'internet', 'cloud', 'firewall', 'protocol',
											 'host', 'hub', 'peer-to-peer', 'router', 'switch', 'server', 'workstation',
											 'bandwidth', 'LAN', 'modem', 'WLAN', 'WAN', 'wireless', 'FTP', 'IP',
											 'port', 'DNS'],
							'talks'       : list()
						},
						'graphics' : {
							'topic_words' : ['graphics', 'bitmap', 'resolution', 'pixel', 'RGB', 'thumbnail', 'compress',
											 'layer', 'crop', 'erase', 'color', 'brightness', 'greyscale', 'frame',
											 'complementary', 'copyright', 'ergonomics'],
							'talks'       : list()
						},
						'databases' : {
							'topic_words' : ['database', 'data', 'table', 'field', 'relational', 'key', 'query',
											 'entity', 'relationship', 'arity', 'constraint', 'DDL', 'SQL',
											 'DBMS', 'record', 'identifier', 'join', 'select', 'check', 'verification'],
							'talks'       : list()
						},
						'computer architecture' : {
							'topic_words' : ['architecture', 'CPU', 'RAM', 'process', 'ALU', 'bus', 'program counter',
											 'clock', 'address', 'memory', 'registers', 'instruction', 'hardware',
											 'volatile', 'operating system', 'von Neumann', 'POST', 'BIOS', 'buffer',
											 'multitasking', 'microprocessor', 'stack pointer', 'fetch', 'pipeling',
											 'intel', 'input', 'output'],
							'talks'       : list()
						},
						'programming languages' : {
							'topic_words' : ['programming', 'debugging', 'interpreters', 'compilers', 'paradigms',
											 'object', 'method', 'class', 'instance', 'code', 'machine', 'Java',
											 'Python', 'C', 'C++', 'Haskell', 'program', 'variables', 'parameter',
											 'pass-by-value', 'value', 'array', 'data structures', 'algorithm',
											 'overload', 'inheritence', 'composition', 'return', 'syntax', 'error',
											 'type', 'alias', 'string', 'int', 'static', 'dynamic', 'pointer'],
							'talks'       : list()
						}
					}
				}
				
		}

		self.model = model

		self.traversal_information = dict()

		self.traverse()



	def traverse(self, location=None, key=None) :
		if location is None : 
			location = self.ontology
		keys =  location.keys()
		if  'topic_words' in keys : 
			self.traversal_information[ key ] = location
			return 
		for key in keys : 
			self.traverse( location[key], key )
		return


	def add_to(self, topic, file_name):
		scores  = [(self.dist_measure(topic, self.traversal_information[key]['topic_words'], key)) for key in self.traversal_information.keys()]
		max_key = sorted(scores, key=lambda tup: tup[0], reverse=True)[0][1]
		self.traversal_information[max_key]['talks'].append(file_name) 

	def dist_measure(self, topic, trav, key):
		#eliminate unnecessary spaces
		topic  = re.sub("\s+"," ",topic)
		topic  = topic.strip()
		tokens = nltk.word_tokenize(topic)
		score  = 0
		count  = 1
		for t in tokens:
			for key_word in trav:
				if(t in self.model and key_word in self.model):
					score = score + self.model.similarity(t, key_word)
					count = count + 1
		return (score/count,key)

	def get_traversal_info(self):
		return self.traversal_information


#given a regex this function will return the value from mail
def find_with_pattern(string, type_list):
	pattern = re.compile(r''+string)
	for t in type_list:
		matches = pattern.findall(t)
		if matches != []:
			return matches[0]
	return None

def get_topic_counter(parsing):
	data       = parsing[0] #tuple type-content
	file_names = parsing[1] #keep track of the name in order to write a tagged mail
							#with the same name

	list_of_words_from_topic = []

	for d in data:

		type_list    = d[0] # list of types
		content_list = d[1] # list of content for each type

		if(len(type_list) == 0):
			continue
		if(find_with_pattern("[Tt]opic", type_list) == None):
			continue

		#find current topic
		topic_pos     = type_list.index(find_with_pattern("[Tt]opic", type_list))
		topic_type    = type_list[topic_pos]
		topic_content = content_list[topic_pos]

		#eliminate unnecessary spaces
		topic_content = re.sub("\s+"," ",topic_content)
		topic_content = topic_content.strip()

		#split it in words
		tokens = nltk.word_tokenize(topic_content)

		#lower case all words to avoid duplicates in counter
		for t in tokens:
			list_of_words_from_topic = list_of_words_from_topic + [t.lower()]

	words_counter = Counter(list_of_words_from_topic)
	return words_counter

def write(created_ontology):
	for key in created_ontology:
		list_of_files = created_ontology[key]['talks']
		for f in list_of_files:
			new_data = ''
			with open('tagged_emails/%s'%f) as current_file:
				for line in current_file:
					pattern = re.compile(r'[Tt]opic')
					matches = pattern.findall(line)
					if(matches != []):
						new_data = new_data + re.sub(r'Topic','Topic<'+key+'>', line) + '\n'
					else:
						new_data = new_data + line + '\n'
			with open('tagged_ontology_emails/%s'%f,'w') as tagged_file:
				tagged_file.write(new_data)

        			



def main():
	#parsing tagged data
	#let user know how to use the program
	try:
		parsing=(parser.parse_data("tagged_emails"))
	except:
		print("Please make sure near this python file you have the following folder format:")
		print("tagged_emails")
		print("Which is automatically generated by seminar_tagger.py")
		return

	print("Tagging e-mails. Please wait...")
	print("This may take some time because of the use of Word2Vec.")

	#create a folder for all txt tagged files named 'tagged_emails'
	if not os.path.exists("tagged_ontology_emails"):
		os.makedirs("tagged_ontology_emails")
	
	#creating the model
	model = Word2Vec.load_word2vec_format( '/home/projects/google-news-corpus/GoogleNews-vectors-negative300.bin', binary=True)

	ontology = Ontology(model)

	data       = parsing[0] #tuple type-content
	file_names = parsing[1] #keep track of the name in order to write a tagged mail
							#with the same name

	for d in data:

		type_list    = d[0] # list of types
		content_list = d[1] # list of content for each type

		if(len(type_list) == 0):
			continue
		if(find_with_pattern("[Tt]opic", type_list) == None):
			continue

		#find current topic
		topic_pos     = type_list.index(find_with_pattern("[Tt]opic", type_list))
		topic_type    = type_list[topic_pos]
		topic_content = content_list[topic_pos]

		ontology.add_to(topic_content, file_names[data.index(d)])

	created_ontology = ontology.get_traversal_info()
	write(created_ontology)

	print('Operation successfully')

main()


