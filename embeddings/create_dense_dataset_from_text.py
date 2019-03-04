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
model = KeyedVectors.load_word2vec_format('skip_s100.txt')
max_dim = (0,0)
dense_dataset = []
#this script should get a dataset which each row is a tuple, with the text and the label and transform it
#in a tuple with the array of the embeddings and the label

#first step should be load the previous dataset
#then should be done some preprocessing to get the embeddings of the sentence
#after getting all the embeddings of the sentences, the arrays should be reshaped with the maximum dimension
emb_dim = 100

#-----------------------------------------------------------
#the preprocessing step also should split the words
def preprocessing(sentence,remove_stopwords=False):
    #translator = str.maketrans('', '', string.punctuation)
    #for i,se in enumerate(sentences):
    se = sentence
    se = se.lower()
    #se = se.translate(string.punctuation)
    #se = se.translate(None, string.punctuation)
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
    #for j in range(len(sentences[i])):
        #sentences[i][j] = sentences[i][j].decode("utf-8")
    #se = se.translate(translator)
    return filtered_sentence


#------------------------------------------------------------
#this function should diferentiate a path to a dataset and a actual dataset array-like
def load_dataset(dataset):

    sentences = dataset[0]
    labels = dataset[1]
    return sentences, labels


#-------------------------------------------------------------
#this function should get the embedding of the sentences
def get_embeddings_concatenated(sentences, labels=None, max_dim=None, remove_stopwords=False):
    for i,sentence in enumerate(sentences):
        # print(sentence)
        sentence = preprocessing(sentence, remove_stopwords=False)
        emb = []
        for w in sentence:
            #emb.append(model.word_vec(str(w).decode("utf-8")))
            try:
                temp  = model.word_vec(str(w))
                # emb.append(model.word_vec(str(w)))

            except:
                #emb.append(np.zeros(emb_dim))
                temp = np.zeros(emb_dim)

            temp = temp.tolist()
            emb += temp
        emb = np.array(emb)
        #row = [emb,label[i]]
        embeddings.append(emb)
        max_dim = max(max_dim, emb.shape)
    return embeddings, max_dim



#---------------------------------------------------------
def get_embeddings(sentences, labels=None, max_dim=None, remove_stopwords=False):
    for i,sentence in enumerate(sentences):
        # print(sentence)
        sentence = preprocessing(sentence, remove_stopwords)
        emb = []
        for w in sentence:
            #emb.append(model.word_vec(str(w).decode("utf-8")))
            try:
                temp  = model.word_vec(str(w))
                # emb.append(model.word_vec(str(w)))

            except:
                #emb.append(np.zeros(emb_dim))
                temp = np.zeros(emb_dim)

            emb.append(temp)
        emb = np.array(emb)
        average = np.average(emb, axis=0)
        average /=(np.linalg.norm(average))
        #row = [emb,label[i]]
        embeddings.append(average)
        max_dim = max(max_dim, emb.shape)
    return embeddings, max_dim


#--------------------------------------------------------------
#as long as the sentences have a different number of words and for computational purposes the shape of them must be equal
#this function will set the default shape as the max of all the sentences, doing some zero padding to all the ones that doesn't fit the shape
def do_padding(embeddings, max_dim, label):
    print("MAXDIM",max_dim)
    data = []
    for i,em in enumerate(embeddings):
        if em.shape != max_dim and em != []:
            print("em shape",em.shape)
            new_row = np.zeros(max_dim[0]).reshape(max_dim)
            new_row[0:em.shape[0]] += em
            new_row.reshape(max_dim[0])
            print("new_row shape ",new_row.shape)

        data.append(new_row.tolist())
        # dense_dataset.append([em,label[i]])
    data = np.array(data)
    label = np.array(label).reshape(data.shape[0],1)
    print("data.shape:",data.shape)
    print("label.shape:",label.shape)
    print(data[1])
    dense_dataset = np.append(data, label, axis=1)
    return dense_dataset

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
        # print("\nline: ",int(line.split("\t")[1]), "\n")
        s = line.split(";")
        print("len",len(s))
        print(s[0])
        data.append(s[0])
        target.append(s[1])
    file.close()
    # data = np.array(data)

    # target = np.array(target)
    dataset = [data,target]
    return dataset

def from_text_to_embeddings(input_path, output_path, method=None):
    #the methods are concatenate, clean_concatenate, avg and clean_avg
    # input_path = sys.argv[1]
    # output_path = sys.argv[2]
    # input_path = "/home/vinicios/reps/hcac/datasets/dilma.tsv"
    # output_path ="/home/vinicios/reps/hcac/datasets/fdilma2.data"
    dataset = load_dataset_from_file(input_path)
    sentences,labels = load_dataset(dataset)
    # sentences = preprocessing(sentences)
    _max_dim = (0,0)
    embeddings = []
    if method == "concatenate":
        emb, max_dim = get_embeddings_concatenated(sentences, max_dim=_max_dim, remove_stopwords=False)
    elif method == "clean_concatenate":
        emb, max_dim = get_embeddings_concatenated(sentences, max_dim=_max_dim, remove_stopwords=True)
    elif method == "avg":
        emb, max_dim = get_embeddings(sentences, max_dim=_max_dim, remove_stopwords=False)
    elif method == "clean_avg":
        emb, max_dim = get_embeddings(sentences, max_dim=_max_dim, remove_stopwords=True)
    else:
        print("ERROR: Invalid method named ",method)
        exit(1)

    #dense_dataset = do_padding(embeddings, max_dim, labels)
    emb = np.array(emb)
    print("embeddings shape",emb.shape)
    label = np.array(labels).reshape(emb.shape[0],1)
    dense_dataset = np.append(emb,label, axis=1)
    emb = None
    return dense_dataset

def save_to_file(dense_dataset, output_path):
    # output_path = "/home/vinicios/reps/hcac/datasets/fdilma2.data"
    file_handler = open(output_path, 'wb')
    pickle.dump(dense_dataset, file_handler)
    # np.savetxt(output, dense_dataset)

if __name__ == '__main__':
    max_dim = (0,0)
    _input_path = sys.argv[1]
    methods = ["avg", "clean_avg"]

    # dense_dataset = np.array(from_text_to_embeddings())
    for m in methods:
        embeddings = []
        print("method: ",m)
        _output_path = "/home/vinicios/hcac/hcac/datasets/skip_s100/"+m+"_tweet2.data"
        dense_dataset = from_text_to_embeddings(input_path=_input_path,output_path=_output_path, method=m)
        save_to_file(dense_dataset,_output_path)
        dense_dataset = []

    # _output_path = "/home/vinicios/reps/hcac/datasets/clean_avg_dilma.data"
    # dense_dataset = from_text_to_embeddings(method="clean_avg",input_path=_input_path,output_path=_output_path)
    # save_to_file(dense_dataset,_output_path)
    #
    # _output_path = "/home/vinicios/reps/hcac/datasets/concatenate_dilma.data"
    # dense_dataset = from_text_to_embeddings(method="concatenate",input_path=_input_path,output_path=_output_path)
    # save_to_file(dense_dataset,_output_path)
    #
    # _output_path = "/home/vinicios/reps/hcac/datasets/avg_dilma.data"
    # dense_dataset = from_text_to_embeddings(method="avg",input_path=_input_path,output_path=_output_path)
    # save_to_file(dense_dataset,_output_path)
