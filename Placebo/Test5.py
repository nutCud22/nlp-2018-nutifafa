import math
import os
from random import sample
import pickle as pickle
from tqdm import tqdm
from scipy.spatial.distance import cdist
import json
import numpy as np
import indicoio
from texttable import Texttable
indicoio.config.api_key = "cbe0522905613dd0a2c4ff6442355e14"



def make_feats(data):
    """
    Send our text data throught the indico API and return each text example's text vector representation
    """
    # TODO
    chunks = [data[x:x+100] for x in range(0, len(data), 100)]
    feats = []

    # just a progress bar to show us how much we have left
    for chunk in chunks:
        feats.extend(indicoio.text_features(chunk))

    return feats


def calculate_distances(feats):
    # cosine distance is the most reasonable metric for comparison of these 300d vectors
    distances = cdist(feats, feats, 'cosine')
    return distances


def similarity_text(idx, distance_matrix, data, faqs, n_similar=5):
    """
    idx: the index of the text we're looking for similar questions to
         (data[idx] corresponds to the actual text we care about)
    distance_matrix: an m by n matrix that stores the distance between
                     document m and document n at distance_matrix[m][n]
    data: a flat list of text data
    """

    # these are the indexes of the texts that are most similar to the text at data[idx]
    # note that this list of 10 elements contains the index of text that we're comparing things to at idx 0
    sorted_distance_idxs = np.argsort(distance_matrix[idx])[:n_similar] # EX: [252, 102, 239, ...]
    # this is the index of the text that is most similar to the query (index 0)
    most_sim_idx = sorted_distance_idxs[1]

    # set the variable that will hold our matching FAQ
    faq_match = None

    for similar_idx in sorted_distance_idxs:
        # actual text data for display
        datum = data[similar_idx]

        # distance in cosine space from our text example to the similar text example
        distance = distance_matrix[idx][similar_idx]

        # how similar that text data is to our input example
        similarity =  1 - distance

        # set a confidence threshold
        # TODO
        if similar_idx == most_sim_idx and similarity >= 0.85:
                    faq_match = data[most_sim_idx]
        else:
            sorry = "Sorry, I'm not sure how to respond."


    # print the appropriate answer to the FAQ, or bring in a human to respond
    # TODO
    if faq_match is not None:
            return faqs[faq_match]
    else:
            return "Sorry, I'm not sure how to respond."

def input_question(question, data, feats):
    # TODO
    # Pass a question

    # add the user question and its vector representations to the corresponding lists, `data` and `feats`
    # insert them at index 0 so you know exactly where they are for later distance calculations
    if question is not None:
        data.insert(0, question)

    new_feats = indicoio.text_features(question)
    feats.insert(0, new_feats)

    return data, feats


def run(test_questions_file):

	# reading files and creating dictionary of questions and answers
	training_questions_file = open("Questions.txt", "r", encoding = "utf-8")
	training_answers_file = open("Answers.txt", "r", encoding = "utf-8")

	training_questions = training_questions_file.readlines()
	training_answers = training_answers_file.readlines()

	faqs  = dict(zip(training_questions, training_answers))

	# TODO
	data = list(faqs.keys()) 

	feats = make_feats(data)

    testing_questions_file = open(test_questions_file, "r", encoding = "utf-8")
    testing_questions = testing_questions_file.readlines()

    results_file = open("qa_results.txt", "wb")

    for question in testing_questions:

    	input_results = input_question(question, data, feats)
    	new_data = input_results[0]
    	new_feats = input_results[1]

    	distance_matrix = calculate_distances(new_feats)

    	idx = 0

    	answer = similarity_text(idx, distance_matrix, new_data, faqs)
    	results_file.write(answer.encode("utf-8"))

run("tester.txt")