
# A cosine similarity calculator for texts
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from pandas import DataFrame as df 

def cosine_sim(first_sen, second_sen): 	
	#passing the first and second sentences as vector variables
	# returning their dot product and magnitudes
	# find cos theta; dot product / product of magnitudes

	dot_product = np.dot(first_sen, second_sen)
	norm_a = np.linalg.norm(first_sen)
	norm_b = np.linalg.norm(second_sen)

	return round(dot_product/(norm_a*norm_b),2)


def count_arrays(input_file):
	file = open(input_file, "r")
	in_file = file.readlines()

	vectorizer = CountVectorizer()
	X = vectorizer.fit_transform(in_file)
	count_array = X.toarray()

	return count_array


def data_frame(count_array):

	length = len(count_array)
	print(list(range(length+1)))

	frame = [[None for _ in range(length)] for _ in range(length)]

	for i in range(length):
		for k in range(length):
			frame[i][k] = cosine_sim(count_array[i], count_array[k])

	for k in frame:
		print(k)

countArray = count_arrays('cos_sim.txt')
data_frame(countArray)