from nltk.corpus import treebank
from nltk.tag import DefaultTagger
from nltk.tag import UnigramTagger
from nltk.tag import BigramTagger
from nltk.tag import TrigramTagger
import nltk
import re
from nltk.tokenize import *	
from nltk.corpus import names
from collections import Counter
import requests
from os import listdir
from os.path import isfile, join
from itertools import combinations
from nltk.corpus.reader import WordListCorpusReader
import parser

#ner tagger class
#started from name tagger Mark gave us
#and improve it to get better results
class NerTagger():
	def __init__(self, training_sents):
		self.training_sents    = training_sents
		#data training from tagged mail + corpus names
		#couldn't find more sadly...
		self.name_training     = names.words() + training_sents['speaker']
		self.location_training = training_sents['location']
		self.counter_names     = Counter(self.name_training) #Counter is way faster than .count() method
		self.counter_loc       = Counter(self.location_training)

	#help when new entity is found
	def split_now(self,list_of_words):
		l = []
		#all singleton lists first
		for l1 in list_of_words:
			l = l + [[l1]]
		#all pairs of 2,3,4... elements
		for start, end in combinations(range(len(list_of_words)), 2):
			l = l + [(list_of_words[start:end+1])]
		return l

	#choose only nnps
	def go_on_with_nnp(self, word_tag_pair):
		list = []
		i    = 0
		while(i< len(word_tag_pair) and word_tag_pair[i][1] == "NNP"):
			list = list + [word_tag_pair[i]]
			i    = i + 1;
		return list;

	#take consecutive group of nnps
	#input: [("asd",NNP),("asd",NNP)...]
	#output: list of strings
	def group_by_nnp(self, word_tag_pair):
		nnp_list = []
		i        = 0 #counter
		while(i < len(word_tag_pair)):
			if(word_tag_pair[i][1]=="NNP"):
				nnp_list = nnp_list + [self.go_on_with_nnp(word_tag_pair[i:])] #returning the list of nnps starting from this element
				i = i + len(self.go_on_with_nnp(word_tag_pair[i:])) # list of lists to keep the successive nnps together
			else:
				i = i + 1;
		return nnp_list



	def tag_it(self, list_of_nnps):
		entity = ""
		for element in list_of_nnps:
			entity = entity + " " + element[0] #construt the entire string

		entity = entity.strip()

		#have we seen it before? if yes it's easy...
		if(entity in self.name_training):
			return(entity, "person")
		if(entity in self.location_training):
			return(entity, "location")

		#split them... maybe it makes more sense
		splitted       = self.split_now(entity.split(" "))
		splitted2      = []
		for s in splitted:
			newstring = " ".join(s)
			splitted2 = splitted2 + [newstring]
		splitted=splitted2

		#which is the most common in our data training
		count_names   = []
		count_loc     = []

		#count for both name and loc
		#if there is add how many times was found
		#if there is not, simply add a 0 to the list
		for s in splitted:
			if s in self.counter_names:
				count_names = count_names + [self.counter_names[s]]
			else:
				count_names = count_names + [0]
			if s in self.counter_loc:
				count_loc   = count_loc   + [self.counter_loc[s]]
			else:
				count_loc   = count_loc   + [0]

		#now check the maximum and decide which is the best solution
		maximum_names = max (count_names)
		maximum_loc   = max (count_loc)

		if(maximum_names != 0 or maximum_loc != 0):
			if(maximum_names >= maximum_loc):
				return(entity, "person")
			else:
				return(entity, "location")

		#try to find on an online encicplopedya
		label_open  = '<Label>'
		label_close = '</Label>'
		pattern_person   = r'' + label_open + r'.*person.*' + r'' + label_close
		pattern_location = r'' + label_open + r'.*location.*' + r'' + label_close

		person_frequency = 0
		loc_frequency    = 0
		#checking for all splitted data
		splitted = [f for f in splitted if len(f)==1] #keep only the singleton to faster the process
		for s in splitted:
			query = requests.get("http://lookup.dbpedia.org/api/search.asmx/KeywordSearch?QueryString=%s"%s).text
			person_found   = re.findall(pattern_person, query)
			loc_found      = re.findall(pattern_location, query)
			if(person_found != []):
				person_frequency = person_frequency + 1
			location_found = re.findall(pattern_location, query)
			if(loc_found != []):
				loc_frequency = loc_frequency + 1

		if(person_frequency==0 and loc_frequency==0):
			return (entity, None)
		if(person_frequency >= loc_frequency):
			return (entity, "person")
		else:
			return (entity, "location")


	#using default pos tagger
	#keep \W characters as it might be useful in splitting nns
	def tag_prop(self, proposition):
		word_tag_pair  = nltk.pos_tag(word_tokenize(proposition))
		grouping       = self.group_by_nnp(word_tag_pair)
		final_list = []
		for g in grouping:
			t = self.tag_it(g)
			final_list = final_list+[t]
		final_list = [f for f in final_list if f[1]!=None]
		return final_list

