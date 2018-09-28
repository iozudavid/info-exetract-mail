import nertagger
import parser
import re
import nltk
from nltk import tokenize
from nltk.tokenize import *	
import os
import sys

#this class will use both parser and nertagger classes

#given a regex this function will return the value from mail
def find_with_pattern(string, type_list):
	pattern = re.compile(r''+string)
	for t in type_list:
		matches = pattern.findall(t)
		if matches != []:
			return matches[0]
	return None	


#simply consider a paragraph by \n\n
def get_paragraphs(content):
	return content.split("\n\n")

#retrieve sentences using nltk sent tockenizer
def get_sentences(paragraphs):
	l = [] 
	for p in paragraphs:
		l = l + [tokenize.sent_tokenize(p)]
	return l

#get time using regex
def get_time(type_list, content_list):
	#find it in header
	pattern_type_time = "([Tt][Ii][Mm][Ee]|[Ww][Hh][Ee][Nn])"
	type_found        = find_with_pattern(pattern_type_time, type_list)
	
	#check if the time tab in header was found
	if(type_found == None):
		return (None, None)	
			
	time_pos = type_list.index(type_found)
	set_time = content_list[time_pos]
	set_time = set_time.strip()
	#all times covered
	time_pattern = "(\d{1,2}\:?(\d{2})?\s*(([AaPp])[Mm])?)"
	only_start_time_pattern = re.compile(r''+time_pattern)
	#both stime and etime: 2 time_patterns devided by '-'
	both_time_pattern = re.compile(r''+time_pattern+r'\s*-\s*'+time_pattern)
	#search for both stime and etime
	if(both_time_pattern.search(set_time)):
		stime_from_header = both_time_pattern.search(set_time).group(1)
		etime_from_header = both_time_pattern.search(set_time).group(5)
		return(stime_from_header, etime_from_header)
	#search for stime only
	if(only_start_time_pattern.search(set_time)):
		stime_from_header = only_start_time_pattern.search(set_time).group()
		return(stime_from_header, None)

	#return None if there is no match for time pattern
	return None

#get speaker using my ner tagger
def get_speaker(type_list, content_list, ner):
	#find it in header
	pattern_type_speaker = "([Ww][Hh][Oo]|[Ss][Pp][Ee][Aa][Kk][Ee][Rr])"
	type_found           = find_with_pattern(pattern_type_speaker, type_list)

	#check if the speaker tab was found
	if(type_found == None):
		return None

	speaker_pos = type_list.index(type_found)
	speaker = content_list[speaker_pos]
	original_speaker = content_list[speaker_pos]
	speaker = speaker.strip()
	#get rid of newlines and tabs
	speaker = re.sub('\n', ' ', speaker)
	speaker = re.sub('\t', ' ', speaker)

	#use the ner tagger I created
	tagged = ner.tag_prop(speaker)

	#if no match return None
	if(tagged == []):
		return None

	tagged = [t for t in tagged if t[1]=='person'] #finding only persons

	#if no person found return None
	if(tagged == []):
		return None

	person = tagged[0] # mostly on first places and also consider only one speaker

	return (''.join(person[0]))


#get location using my ner tagger
def get_location(type_list, content_list, ner):
	#find it in header
	pattern_type_location = "([Ww][Hh][Ee][Rr][Ee]|[Pp][Ll][Aa][Cc][Ee]|[Ll][Oo][Cc][Aa][Tt][Ii][Oo][Nn]|[Pp][Ll][Aa][Cc][Ee])"
	type_found            = find_with_pattern(pattern_type_location, type_list)

	#if there is no location tab in header, abandon...
	if(type_found == None):
		return None

	loc_pos = type_list.index(type_found)
	loc = content_list[loc_pos]
	original_loc = loc
	loc = loc.strip()

	#get rid of newlines and tabs
	loc = re.sub('\n', ' ', loc)
	loc = re.sub('\t', ' ', loc)

	#using ner tagger I created
	tagged = ner.tag_prop(loc)

	#if there are is no match abandon
	if(tagged == []):
		return None

	tagged   = [t for t in tagged if t[1]=='location'] #finding only locations

	#if no location, abandon...
	if(tagged == []):
		return None

	location = tagged[0] #assuming it's the first match

	return (''.join(location[0]))

#first action in rebuild the abstract content
def tag_paragraphs_and_sentences(sentences_from_paragraphs):		
	abstract_tagged = ""
	for sentences in sentences_from_paragraphs:
		abstract_tagged = abstract_tagged + "<paragraph>"
		for s in sentences:
			abstract_tagged = abstract_tagged + "<sentence>"
			abstract_tagged = abstract_tagged + s[:-1]
			abstract_tagged = abstract_tagged + "</sentence>"
			abstract_tagged = abstract_tagged + s[-1]
		abstract_tagged = abstract_tagged + "</paragraph>\n"
	return abstract_tagged


#given a tag and a pattern, replace all matches with <tag>match</tag>
#this will tag if the content matches exactly the header
#used only for speaker and location
def tag_email(content_list, key_search, tag1, tag2, content):
	to_search = re.compile(key_search)
	found = to_search.findall(content)
	if(found!=[]):
		cont_index = content_list.index(content)
		content = re.sub(found[0],tag1+found[0]+tag2,content)
		content_list[cont_index] = content
	return content_list

#finally write the tagged e-mails
def write(file_name, type_list, content_list):
	file = open("tagged_emails/%s"%file_name, 'w')
	for i in range(len(content_list)):
		if(type_list[i] == ""):
			#for unique tag of all emails eg:<0.1...>
			file.write((content_list[i])+"\n")
		else:
			file.write(type_list[i] + ":" + content_list[i] + "\n")

#used if no speaker in header
#return a tuple containing pattern before speaker and after speaker
def get_pattern_for_speaker():
	pattern_before = ['presented by', 'introduced by', 'taught by', 'held by',
					'lecture by', 'given by', 'speaker is', 'talk by', 'served by']

	pattern_after  = ['will present', 'is presenting', 'will introduc', 'is indroducing',
	                 'will teach', 'is teaching', 'will hold', 'is holding', 
	                 'will talk', 'is talking', 'will speak', 'is speaking']
	return (pattern_before, pattern_after)

#last opportunity
#find it with a predefined pattern
def try_to_find_speaker_in_abstract(abstract_content):
	abstract_tokenzied = word_tokenize(abstract_content)
	abstract_tagged    = nltk.pos_tag(abstract_tokenzied)
	pattern_before     = get_pattern_for_speaker()[0]
	pattern_after      = get_pattern_for_speaker()[1]
	for element in abstract_tagged:
		word = element[0]
		tag  = element[1]
		pos  = abstract_tagged.index(element)
		if( (pos+1) < len(abstract_tagged) and (word + ' ' + abstract_tagged[pos+1][0]) in pattern_before):
			if(pos+2 < len(abstract_tagged)):
				first_name_word  = abstract_tagged[pos+2][0]
				first_name_tag   = abstract_tagged[pos+2][1]
				second_name_word = abstract_tagged[pos+3][0]
				second_name_tag  = abstract_tagged[pos+3][1]
				if(first_name_tag == 'NNP' and second_name_tag == 'NNP'):
					speaker_from_header = first_name_word + ' ' + second_name_word
					return speaker_from_header
		if( (pos+1) < len(abstract_tagged) and (word + ' ' + abstract_tagged[pos+1][0]) in pattern_after):
			first_name_word  = abstract_tagged[pos-1][0]
			first_name_tag   = abstract_tagged[pos-1][1]
			second_name_word = abstract_tagged[pos-2][0]
			second_name_tag  = abstract_tagged[pos-2][1]
			if(first_name_tag == 'NNP' and second_name_tag == 'NNP'):
				speaker_from_header = first_name_word + ' ' + second_name_word
				return speaker_from_header
	return None

def main():

	#parsing both untagged and tagged
	#let user know how to use the program
	try:
		parsing=(parser.parse_data("nltk_data/corpora/untagged"))
		data_tagged=(parser.parse_data("nltk_data/corpora/training"))
	except:
		print("Please make sure near this python file you have the following folder format:")
		print("\"nltk_data/corpora/untagged\" and \"nltk_data/corpora/training\"")
		print("In order to access tagged and untagged emails")
		return

	print("Tagged and untagged e-mails have been read.")
	print("Tagging e-mails. Please wait...")
	#print("This might take a while because of requests to enciclopedya...")
	#set up the time,location,speaker dict from tagged
	train_sents = parser.get_training_sents(data_tagged[0])

	#create a folder for all txt tagged files named 'tagged_emails'
	if not os.path.exists("tagged_emails"):
		os.makedirs("tagged_emails")
	
	data       = parsing[0] #tuple type-content
	file_names = parsing[1] #keep track of the name in order to write a tagged mail
							#with the same name

	#first train the tagger and initialise it only once for all emails
	ner = nertagger.NerTagger(train_sents)

	for d in data:
		
		type_list    = d[0] # list of types
		content_list = d[1] # list of content for each type

		#continue if no content is found
		if(len(type_list) == 0):
			continue
		if(find_with_pattern("[Aa]bstract", type_list) == None):
			continue

		#find current abstract
		abstract_pos     = type_list.index(find_with_pattern("[Aa]bstract", type_list))
		abstract_type    = type_list[abstract_pos]
		abstract_content = content_list[abstract_pos]

		#split into paragraphs and sentences
		paragraphs = get_paragraphs(abstract_content)	
		sentences_from_paragraphs = get_sentences(paragraphs)

		#there are always useful informations in header
		#deal with them first
		stime_from_header    = get_time(type_list, content_list)[0]
		etime_from_header    = get_time(type_list, content_list)[1]
		speaker_from_header  = get_speaker(type_list, content_list, ner)
		location_from_header = get_location(type_list, content_list, ner)

		if(speaker_from_header  != None):
			speaker_from_header  = speaker_from_header.strip()
		if(location_from_header != None):
			location_from_header = location_from_header.strip()

		if(speaker_from_header == None or speaker_from_header == ""):
			speaker_from_header = try_to_find_speaker_in_abstract(abstract_content)

		#setting up the tagged corpus
		abstract_tagged = tag_paragraphs_and_sentences(sentences_from_paragraphs)
		content_list[abstract_pos] = abstract_tagged

		for content in content_list:
			if(stime_from_header != None   and stime_from_header!=""):
				tag_email(content_list, stime_from_header,   "<stime>",   "</stime>",   content)
		for content in content_list:
			if(etime_from_header != None   and etime_from_header!=""):
				tag_email(content_list, etime_from_header,   "<etime>",   "</etime>",   content)
		for content in content_list:	
			if(speaker_from_header != None and speaker_from_header != ""):
				tag_email(content_list, speaker_from_header, "<speaker>", "</speaker>", content)
		for content in content_list:	
			if(location_from_header != None and location_from_header != "" and "*" not in location_from_header): #had some troubles with * inputs
				tag_email(content_list, location_from_header, "<location>", "</location>", content)

		


		#get file name and write
		file_name = file_names[data.index(d)]
		write(file_name, type_list, content_list)

	print("Operation successful.")
	print("Tagged files should be in "+os.getcwd()+"/tagged_emails")

main()


