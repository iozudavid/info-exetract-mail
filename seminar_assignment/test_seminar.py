from os import listdir
from os.path import isfile, join
from itertools import combinations
from nltk.corpus.reader import WordListCorpusReader
import re
from collections import Counter

my_tagged_path    = 'tagged_emails'
test_tagged_path  = 'test_tagged'

list_of_patterns = ['<speaker>(.*?)</speaker>',
					'<stime>(.*?)</stime>',
					'<etime>(.*?)</etime>',
					'<location>(.*?)</location>',
					'<sentence>(.*?)</sentence>']

def get_file_paths(path=''):
	only_files = [f for f in listdir(path) if isfile(join(path,f)) and "txt" in f]
	return only_files

my_files   = get_file_paths(my_tagged_path)
test_files = get_file_paths(test_tagged_path)

true_positive  = {'speaker'  : 1,
			   'stime'    : 1,
			   'etime'    : 1,
			   'location' : 1,
			   'sentence' : 1}

true_negative = {'speaker'  : 1,
			   'stime'    : 1,
			   'etime'    : 1,
			   'location' : 1,
			   'sentence' : 1}

false_positive = {'speaker'  : 1,
			   'stime'    : 1,
			   'etime'    : 1,
			   'location' : 1,
			   'sentence' : 1}

false_negative = {'speaker'  : 1,
			   'stime'    : 1,
			   'etime'    : 1,
			   'location' : 1,
			   'sentence' : 1}

for f in test_files:

	data_test = open (test_tagged_path+"/"+f, "r").read()
	my_data   = open (my_tagged_path+"/"+f, "r").read()

	for p in list_of_patterns:
		pat = '<(.*?)>'
		c = re.compile(pat)
		fa = c.findall(p)
		what = fa[0]

		comp = re.compile(p, re.DOTALL)
		found_test = comp.findall(data_test)
		my_found   = comp.findall(my_data)

		pat2 = '<.*?>'
		for f in found_test:
			index = found_test.index(f)
			f = re.sub(pat2,"",f)
			found_test[index] = f

		for f in my_found:
			index = my_found.index(f)
			f = re.sub(pat2,"",f)
			my_found[index] = f


		my_counter = Counter(my_found)
		ideal_counter = Counter(found_test) 

		if(len(my_counter)==1 and len(ideal_counter)==1):
			a=true_negative[what]
			true_negative[what] = a + 1

		for m in ideal_counter:
			if(m in my_counter):
				a=true_positive[what]
				true_positive[what]=a+my_counter[m]
				a=false_negative[what]
				false_negative[what] = a + (ideal_counter[m] - my_counter[m])
			else:
				a=false_negative[what]
				false_negative[what]=a+ideal_counter[m]
		for m in my_counter:
			my_val = my_counter[m]
			if(m not in ideal_counter):
				a = false_positive[what]
				false_positive[what] = a + my_val

print('Writing results in seminar_tagger_results.txt')
file = open("seminar_tagger_results", 'w')

file.write("\n==========Precision==========\n")
for g in true_positive:
	precision=(true_positive[g]/(float(true_positive[g]+false_positive[g])))
	file.write(g + ": " + str(precision) + "\n")

file.write("\n==========Recall==========\n")
for g in true_positive:
	recall=(true_positive[g]/(float(true_positive[g])+false_negative[g]))
	file.write(g + ": " + str(recall) + "\n")

file.write("\n==========F1 Measure==========\n")
for g in true_positive:
	precision=(true_positive[g]/(float(true_positive[g]+false_positive[g])))
	recall=(true_positive[g]/(float(true_positive[g])+false_negative[g]))    
	f1_measure=(2*precision*recall/(float(precision + recall)))
	file.write(g + ": " + str(f1_measure) + "\n")

file.write("\n==========Accuracy==========\n")
for g in true_positive:
	accuracy=((true_positive[g]+true_negative[g])/(float(true_positive[g]+true_negative[g]+false_positive[g]+false_negative[g])))
	file.write(g + ": " + str(accuracy) + "\n")


print('Operation done.')




