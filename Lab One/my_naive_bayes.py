
# coding: utf-8

# In[1]:


import random
import sys


# In[2]:


def prep_data(*files):			#loading the data from the files 
	data = []
	
	for file in files:
		in_file = open(file, 'rt')
		text = in_file.readlines()
		in_file.close()
		
		for line in text:
			#spliting text by whitespaces
			sentence = line.split()
			
			#shrinking vocab size by lowercase
			sentence = [word.lower() for word in sentence]
			
			#using string library to remove punctuations
			#as used by Jason Brownlee
			import string 
			table = str.maketrans('', '', string.punctuation)
			sentence = [word.translate(table) for word in sentence]
			data.append(sentence)

	return data


# In[8]:


class NaiveBayes:

	def __init__(self,files, percentile):
		self.data1 = None
		self.training_set = None
		self.test_set = None
		self.document = None
		self.class_list = None
		self.log_priors = None
		self.loglikelihood = None
		self.voc_doc = None
		self.predictions = None
		self.accuracy = None
		self.files = files
		self.split_percentile = percentile
	
	def load_data(self):			#loading the data from the files 
		data = []
		
		for file in self.files:
			in_file = open(file, 'rt')
			text = in_file.readlines()
			in_file.close()
			
			for line in text:
				#spliting text by whitespaces
				sentence = line.split()
				
				#shrinking vocab size by lowercase
				sentence = [word.lower() for word in sentence]
				
				#using string library to remove punctuations
				#as used by Jason Brownlee
				import string 
				table = str.maketrans('', '', string.punctuation)
				sentence = [word.translate(table) for word in sentence]
				data.append(sentence)

		self.data1 = data
		

	
	def spliting_dataset(self):			#spliting data into training and test data
		
		copy = self.data1
		training_data_size = int(len(copy) * self.split_percentile)
		training_set = []
		
		while len(training_set) < training_data_size:
			index = random.randrange(len(copy))
			training_set.append(copy.pop(index)) 
		
		#after appending to the training set, the remaining items in data will be used for testing
		#data being returned is the test data
		self.training_set = training_set
		self.test_set = copy
		

	def spliting_by_class(self):
		
		class_separation = {}
		for i in range(len(self.data1)):
			sentence = self.data1[i]
			if sentence[-1] not in class_separation:
				class_separation[sentence[-1]] = []
			class_separation[sentence[-1]].append(sentence[:-1])
		
		self.document = class_separation
		self.class_list = class_separation.keys()
		


	def train_naive_bayes(self):			#training naive bayes
														#training document passed is a dictionary
		
		log_priors = []
		loglikelihood =  {}
		num_doc = sum([len(self.document[tense]) for tense in self.class_list])
		
		# Creating vocabulary of all words in document
		voc_doc = set()
		
		for class_given in self.class_list:
			for tenses in self.document[class_given]:
				for word in tenses:
					voc_doc.add(word)

		import math
		from collections import Counter

		for class_given in self.class_list:
			loglikelihood[class_given] = []
			num_class = len(self.document[class_given])
		
			logprior_class_given = math.log(num_class/num_doc)
			
			log_priors.append([class_given, round(logprior_class_given,4)])
			
			#creating list of words in given class
			bigdoc_class = []
			for tenses in self.document[class_given]:
				for word in tenses:
					bigdoc_class.append(word)
			
			wordcount = Counter(bigdoc_class)
						
			for word in voc_doc:
				count_in_class_given = wordcount[word]
				
				loglikelihood_in_class_given_numerator = count_in_class_given + 1
				loglikelihood_in_class_given_denominator = len(bigdoc_class) + len(voc_doc)
				loglikelihood_in_class_given = math.log(loglikelihood_in_class_given_numerator/ loglikelihood_in_class_given_denominator)
				
				# holding loglikelihoods of each word in each class in a list with index same as class value
				loglikelihood[class_given].append([word, round(loglikelihood_in_class_given,4)])			
				
		self.log_priors = log_priors
		self.loglikelihood = loglikelihood
		self.voc_doc = voc_doc


	def retrieve_loglikelihood(self, loglikelihood_dict,word,class_given):
		likelihood = 0
		list = loglikelihood_dict[class_given]
		for bond in list:
			if bond[0] == word:
				likelihood = bond[1]
		return likelihood


	def retrieve_logprior(self, logprior_list, class_given):
		prior = 0
		for list in logprior_list:
			if list[0] == class_given:
				prior = list[1]
		return prior

	def train(self):
		self.load_data()
		
		self.spliting_dataset()
		
		self.spliting_by_class()
		
		self.train_naive_bayes()


	def getTestSet(self):
		return self.test_set

	def getPredictions(self):
		return self.predictions

	def getAccuracy(self):
		return self.accuracy

	def test_naive_bayes(self, test_set):
		argmax_by_class = []
		predictions = []
		for class_given in self.class_list:
			for tenses in test_set:
				sum_class_given = self.retrieve_logprior(self.log_priors, class_given)
				#print(tenses)
				for word in tenses[:-1]:
					if word in self.voc_doc:
						sum_class_given += self.retrieve_loglikelihood(self.loglikelihood,word,class_given)
				
				argmax_by_class.append([class_given,sum_class_given])
		
		#retrieving best predicted class
		for i in range(int((len(argmax_by_class)/2))):
			if argmax_by_class[i][1] > argmax_by_class[i+int((len(argmax_by_class)/2))][1]:
				predictions.append(argmax_by_class[i][0])
			else:
				predictions.append(argmax_by_class[i+int((len(argmax_by_class)/2))][0])
		
		self.predictions = predictions

	def check_accuracy(self, test_set):
		correct = 0
		for i in range(len(test_set)):
			if test_set[i][-1] == self.predictions[i]:
				correct += 1
		self.accuracy = round((correct/float(len(test_set))) * 100.0, 2)
    
if __name__ == '__main__':
    model = NaiveBayes(['amazon_cells_labelled.txt', 'yelp_labelled.txt', 'imdb_labelled.txt'], 0.9)
    model.train()
    test_file = sys.argv[1]
    test_set = prep_data(test_file)
    model.test_naive_bayes(test_set)
    predictions = model.getPredictions()

    out_file = open('results_file.txt', 'w')
    for prediction in predictions:
        out_file.write(prediction + "\n")


    model.check_accuracy(test_set)
    accuracy = model.getAccuracy()
    out_file.write(str(accuracy) + "% accuracy")

    out_file.close()

