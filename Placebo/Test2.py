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
indicoio.config.api_key = "edd63dd43797ccb9cb13606856193622"


def make_feats(data):
    """
    Send our text data throught the indico API and return each text example's text vector representation
    """
    # TODO
    chunks = [data[x:x+100] for x in range(0, len(data), 100)]
    feats = []

    # just a progress bar to show us how much we have left
    for chunk in tqdm(chunks):
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
    t = Texttable()
    t.set_cols_width([50, 20])

    # these are the indexes of the texts that are most similar to the text at data[idx]
    # note that this list of 10 elements contains the index of text that we're comparing things to at idx 0
    sorted_distance_idxs = np.argsort(distance_matrix[idx])[:n_similar] # EX: [252, 102, 239, ...]
    # this is the index of the text that is most similar to the query (index 0)
    most_sim_idx = sorted_distance_idxs[1]

    # header for texttable
    t.add_rows([['Text', 'Similarity']])
    print(t.draw())

    # set the variable that will hold our matching FAQ
    faq_match = None

    for similar_idx in sorted_distance_idxs:
        # actual text data for display
        datum = data[similar_idx]

        # distance in cosine space from our text example to the similar text example
        distance = distance_matrix[idx][similar_idx]

        # how similar that text data is to our input example
        similarity =  1 - distance

        # add the text + the floating point similarity value to our Texttable() object for display
        t.add_rows([[datum, str(round(similarity, 2))]])
        print(t.draw())

        # set a confidence threshold
        # TODO
        if similar_idx == most_sim_idx and similarity >= 0.75:
                    faq_match = data[most_sim_idx]
        else:
            sorry = "Sorry, I'm not sure how to respond. Let me find someone who can help you."


    # print the appropriate answer to the FAQ, or bring in a human to respond
    # TODO
    if faq_match is not None:
            print ("A: %r" % faqs[faq_match])
    else:
            print ("sorry")

def input_question(data, feats):
    # TODO
    # input a question
    question = input("What is your question? ")

    # add the user question and its vector representations to the corresponding lists, `data` and `feats`
    # insert them at index 0 so you know exactly where they are for later distance calculations
    if question is not None:
        data.insert(0, question)

    new_feats = indicoio.text_features(question)
    feats.insert(0, new_feats)

    return data, feats


def run():

    # # reading files and creating dictionary of questions and answers
    # questions_file = open("Questions.txt", "r", encoding = "utf-8")
    # answers_file = open("Answers.txt", "r", encoding = "utf-8")

    # input_questions = questions_file.readlines()
    # input_answers = answers_file.readlines()

    # faqs  = dict(zip(input_questions, input_answers))


    # TODO
    data = list(faqs.keys()) 
    print ("FAQ data received. Finding features.")

    feats = make_feats(data)

    with open('faq_feats.pkl', 'wb') as f:
        pickle.dump(feats, f)
    print ("FAQ features found!")

    with open('faq_feats.pkl', 'rb') as f:
        feats = pickle.load(f)
    print ("Features found -- success! Calculating similarities...")

    input_results = input_question(data, feats)
    new_data = input_results[0]
    new_feats = input_results[1]

    distance_matrix = calculate_distances(new_feats)
    print ("Similarities found. Generating table.")

    idx = 0
    similarity_text(idx, distance_matrix, new_data, faqs)
    print ('\n' + '-' * 80)

if __name__ == "__main__":
    run()