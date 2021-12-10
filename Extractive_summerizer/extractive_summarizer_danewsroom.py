# from https://towardsdatascience.com/understand-text-summarization-and-create-your-own-summarizer-in-python-b26a9f09fc70


# coding: utf-8
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords # Also contains stopwords in Danish
# A stopword is e.g. "and", "me", "can"
# check with print(stopwords.words('english'))
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx # tool for graphs
 
#filedata = chunk['text'][3]
import re
# splits a txt file into sentences that are also tokenized
def prepare_article(file):
    filedata = file.replace("\n\n", ". ")
    article = filedata.split(". ")
    #article = filedata.split("\n\n") # I added this since it is in many of the articles instead of .
    sentences = []

    for sentence in article:
        print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop() 
   
    
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(file_name, top_n=5):
    stop_words = stopwords.words('danish')
    summarize_text = []

    # Step 1 - Read text anc split it
    sentences = prepare_article(file_name)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph, max_iter=1000)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    #print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))


    # Step 5 - Offcourse, output the summarize texr
    print("Summarize Text: \n", ". ".join(summarize_text))

    return summarize_text


# Making it for the danewsroom dataset - problem with the iterate
import pandas as pd

# Rouge scores
from rouge_score import rouge_scorer


scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

# Lists for output
rouge_scores = []
summaries = []
filesummaries = []

import csv
#df = pd.read_csv('../../NP Exam/danewsroom.csv', chunksize=10000, iterator=True)
df = pd.read_csv('../../NP Exam/danewsroom.csv', nrows=100)

def summarise_danewsroom(df):
    for iter_num in range(len(df)):
        if iter_num != 29:
            # do things with chunk
            filedata = df['text'][iter_num]
            filesummary = df['summary'][iter_num]
            

            summary = generate_summary(filedata, 1) 
            summary = " ".join(map(str, summary)) # from list of sentences to a string object
                
            # Rouge scores
            scores = scorer.score(filesummary, summary)
                
            rouge_scores.append(scores)
            filesummaries.append(filesummary)
            summaries.append(summary)
        else:
            print("moving on")
            continue
        
        # break
        if iter_num == 100:
            break 
    return list([rouge_scores, filesummaries, summaries])


output = summarise_danewsroom(df)


