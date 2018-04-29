#Aishwarya Sudhakar - Feb 6, 2018
import csv
import codecs
import string
from io import open
#import nltk
#from nltk.corpus import words
#from nltk.corpus import stopwords
from random import shuffle
import numpy as np
import re
from sklearn import preprocessing

def rm_punc(str):
	return re.sub("[^a-zA-Z0-9 ]", "", str)

def cleantext(input_text):

	#inter_text = "".join((char if  re.match('^[a-zA-Z0-9]+$',char) else " ") for char in input_text).split()
	wordset = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])
	output_text = [w for w in input_text.split() if w not in wordset and len(w)>2]
	
	return " ".join(output_text)

#Parse CSV file -- Returns two lists, tokenized x data, and normalized y value
def parse_metadata_bugs_analysts(csv_name):
		print("Opening file: "+csv_name)
		all_data = [];
		all_owner = [];
		#with codecs.open(csv_name, 'r', encoding="utf-8") as file:
		with open(csv_name, 'rb') as file:
			reader = csv.DictReader(file)
			for row in reader:
				#Remove the bugzilla assignee - this is the assignment to system which is about 500/10000 bugs
				label = row['Assignee']
				summary = rm_punc(row['Summary']) 
				if len(summary.split())>4:
					summary = rm_punc(row['Product']) + ' ' + rm_punc(row['Component']) + ' ' + rm_punc(row['Keywords'] + summary)
					lc_summary = summary.lower().split()
					#new_summary = ' '.join(lc_summary)
					if label != "bugzilla":
						all_data.append(lc_summary)
						all_owner.append(label)

		min_len, max_len = get_stats(all_data, all_owner)
		print("Min length of sentence", min_len)
		print("Max length of sentence", max_len)

		#print 'shape all data', np.array(all_data).shape()
		#print 'all data[1,1] ', all_data[1,1] 
		return np.array(all_data), np.array(all_owner), min_len, max_len



def get_stats(all_data, all_owner):

	min_len = 100
	for one_data in all_data:
		min_len = min(min_len, len(one_data))

	max_len = 0
	for one_data in all_data:
		max_len = max(max_len, len(one_data))

	return min_len, max_len

all_data, all_owner, min_sentence_length, max_sentence_len = parse_metadata_bugs_analysts('bugs-2018-02-09.csv')

print(type(all_data))