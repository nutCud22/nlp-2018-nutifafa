from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm



def feature_extraction(data):
	'''
		Returning each examples text vectorized form
		Batch size of 100 for each tokenization
	'''
	batch_size = 100
	batches = [data[x:x+batch_size] for x in range(0, len(data), batch_size)]

	'''
		list = [1,2,3,4,5,6,7,8,9,0]
		batch = [list[a:a+2] for a in range(0,len(list), 2)]
		>>> [[1, 2], [3, 4], [5, 6], [7, 8], [9, 0]]
	'''
	features = []

	unigram_vectorizer = CountVectorizer(ngram_range = (1,1))
	analyzer = unigram_vectorizer.build_analyzer()

	'''
		Using a prograss bar to show progress of batching progress
	'''
	for batch in tqdm(batches):
		print(batch)
		
		features.extend(analyzer(batch))


	print(features)



def run():

	faqs = {
    'Where can I find my API Key?':'Hi there! You receive an API key upon sign up. After you confirm your email you will be able to log in to your dashboard at indico.io/dashboard and see your API key on the top of the screen.',
    'Can indico be downloaded as a package and used offline?':'Unfortunately, no. However we do have a paid option for on premise deployment for enterprise clients.',
    'What is indico API credit?':'Hello! indico API credit is what we use to keep track of usage. If you send in 100 bits of text into our API you are charged 100 credits, essentially one credit is consumed per datapoint analyzed. Every user gets 10,000 free API credits per month.',
    'Would I be able to set up a Pay as You Go account and have it stop if I reach 10,000 calls so that I won\'t be charged if I accidentally go over the limit?':'Hi there! Yep, the best way for you to do this would be to sign up for a pay as you go account and don\'t put in a credit card (we don\'t require you to). When you hit 10,000 you will be locked out of your account and unable to make more calls until you put a credit card in or you can wait until the first of the month when it resets to 10,000.',
    'Hello! When I try to install indico with pip, I get this error on Windows. Do you know why?':'Hello, please try following the steps listed here: https://indico.io/blog/getting-started-indico-tutorial-for-beginning-programmers/#windows and let us know if you still continue to have problems.'
	}

    # reading files and creating dictionary of questions and answers
    # questions_file = open("Questions.txt", "r", encoding = "utf-8")
    # answers_file = open("Answers.txt", "r", encoding = "utf-8")

    # input_questions = questions_file.readlines()
    # input_answers = answers_file.readlines()

    # faqs  = dict(zip(input_questions, input_answers))


    # TODO
	data = list(faqs.keys())
	print ("FAQ data received. Finding features.")

	feature_extraction(data)


run()