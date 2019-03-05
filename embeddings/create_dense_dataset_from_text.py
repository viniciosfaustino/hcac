#!/usr/bin/env python
# coding: utf-8

from gensim.models import KeyedVectors
import numpy as np
import string
import sys
import re
import csv
import pickle
from nltk.corpus import stopwords
import os

dense_dataset = []
#this script should get a dataset which each row is a tuple, with the text and the label and transform it
#in a tuple with the array of the embeddings and the label

#first step should be load the previous dataset
#then should be done some preprocessing to get the embeddings of the sentence
#the new dataset will be composed by the average of the embeddigns of the sentence, normalized by the norm of the average embeddings array

#-----------------------------------------------------------
#the preprocessing step also should split the words
def preprocessing(sentence,remove_stopwords=False):

    se = sentence
    se = se.lower()

    se = re.sub('['+string.punctuation+']', '', se)
    se = se.replace(".","")
    se = " ".join(se.split())
    sentence = se.split()
    filtered_sentence = []
    if remove_stopwords:
        stop_words = set(stopwords.words("portuguese"))
        filtered_sentence = [w for w in sentence if not w in stop_words]
    else:
        filtered_sentence = sentence

    return filtered_sentence


#------------------------------------------------------------
#this function should diferentiate a path to a dataset and a actual dataset array-like
def load_dataset(dataset):
    sentences = dataset[0]
    labels = dataset[1]
    return sentences, labels


#-------------------------------------------------------------
#this function should get the embedding of the sentences
def get_embeddings_concatenated(sentences, labels=None, remove_stopwords=False):
    for i,sentence in enumerate(sentences):
        sentence = preprocessing(sentence, remove_stopwords=False)
        emb = []
        for w in sentence:
            try:
                temp  = model.word_vec(str(w))

            except:
                temp = np.zeros(emb_dim)

            temp = temp.tolist()
            emb += temp
        emb = np.array(emb)

        embeddings.append(emb)
    return embeddings



#---------------------------------------------------------
def get_embeddings(sentences, labels=None, remove_stopwords=False):
    for i,sentence in enumerate(sentences):
        sentence = preprocessing(sentence, remove_stopwords)
        emb = []
        for w in sentence:
            try:
                temp  = model.word_vec(str(w))

            except:
                temp = np.zeros(emb_dim)

            emb.append(temp)
        emb = np.array(emb)
        average = np.average(emb, axis=0)
        average /=(np.linalg.norm(average))

        embeddings.append(average)
    return embeddings

#------------------------------------------------------------
#this function should get the dataset from a file specified in the function calling or as an argument of the script
def load_dataset_from_file(path):
    if path != None:
        input = path
    else:
        input = sys.argv[1]

    data = []
    target = []
    file = open(input, "r")
    for line in file:
        s = line.split(";")
        data.append(s[0])
        target.append(s[1])
    file.close()

    dataset = [data,target]
    return dataset

def from_text_to_embeddings(input_path, output_path, method=None):
    #the methods are standart (std), and no_stopwords
    dataset = load_dataset_from_file(input_path)
    sentences,labels = load_dataset(dataset)

    embeddings = []
    if method == "std":
        emb = get_embeddings(sentences, remove_stopwords=False)
    elif method == "no_stopwords":
        emb = get_embeddings(sentences, remove_stopwords=True)
    else:
        print("ERROR: Invalid method named ",method)
        exit(1)

    emb = np.array(emb)
    print("embeddings shape",emb.shape)
    label = np.array(labels).reshape(emb.shape[0],1)
    dense_dataset = np.append(emb,label, axis=1)
    emb = None
    return dense_dataset

def save_to_file(dense_dataset, output_path):
    file_handler = open(output_path, 'wb')
    pickle.dump(dense_dataset, file_handler)

if __name__ == '__main__':
    _input_path = sys.argv[1]
    methods = ["std", "no_stopwords"]
    pretrained_embeddings = ["cbow_s50", "cbow_s100"]

    dataset_name = os.path.splitext(os.path.basename(_input_path))[0]

    for pe in pretrained_embeddings:
        model = KeyedVectors.load_word2vec_format(pe+'.txt')
        for method in methods:
            print("Generating dataset from "+pe+" and "+ method)
            emb_dim = model.vector_size
            embeddings = []
            print("method: ",method)
            _output_path = "/home/vinicios/hcac/hcac/datasets/"+pe+"_"+method+"_"+dataset_name+".data"
            dense_dataset = from_text_to_embeddings(input_path=_input_path,output_path=_output_path, method=method)
            save_to_file(dense_dataset,_output_path)
            dense_dataset = []
