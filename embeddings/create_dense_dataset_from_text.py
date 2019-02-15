#!/usr/bin/env python
# coding: utf-8

# In[10]:



from gensim.models import KeyedVectors
import numpy as np
import string
import sys
import re
import csv
import pickle
from nltk.corpus import stopwords
model = KeyedVectors.load_word2vec_format('cbow_s50.txt')
max_dim = (0,0)
dense_dataset = []
#this script should get a dataset which each row is a tuple, with the text and the label and transform it
#in a tuple with the array of the embeddings and the label

#first step should be load the previous dataset
#then should be done some preprocessing to get the embeddings of the sentence
#after getting all the embeddings of the sentences, the arrays should be reshaped with the maximum dimension


#-----------------------------------------------------------
#the preprocessing step also should split the words
def preprocessing(sentence):
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
    stop_words = set(stopwords.words("portuguese"))
    filtered_sentence = [w for w in sentence if not w in stop_words]
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
def get_embeddings_concatenated(sentences, labels=None, max_dim=None):
    for i,sentence in enumerate(sentences):
        # print(sentence)
        sentence = preprocessing(sentence)
        emb = []
        for w in sentence:
            #emb.append(model.word_vec(str(w).decode("utf-8")))
            try:
                temp  = model.word_vec(str(w))
                # emb.append(model.word_vec(str(w)))

            except:
                #emb.append(np.zeros(50))
                temp = np.zeros(50)

            temp = temp.tolist()
            emb += temp
        emb = np.array(emb)
        #row = [emb,label[i]]
        embeddings.append(emb)
        max_dim = max(max_dim, emb.shape)
    return embeddings, max_dim



#---------------------------------------------------------
def get_embeddings(sentences, labels=None, max_dim=None):
    for i,sentence in enumerate(sentences):
        # print(sentence)
        sentence = preprocessing(sentence)
        emb = []
        for w in sentence:
            #emb.append(model.word_vec(str(w).decode("utf-8")))
            try:
                temp  = model.word_vec(str(w))
                # emb.append(model.word_vec(str(w)))

            except:
                #emb.append(np.zeros(50))
                temp = np.zeros(50)

            emb.append(temp)
        emb = np.array(emb)
        average = np.average(emb, axis=0)
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
        data.append(str(line.split("\t")[0]))
        target.append(int(line.split("\t")[1]))
    file.close()
    # data = np.array(data)

    # target = np.array(target)
    dataset = [data,target]
    return dataset

def from_text_to_embeddings():
    # input_path = sys.argv[1]
    # output_path = sys.argv[2]
    input_path = "/home/vinicios/reps/hcac/datasets/brasui2.csv"
    output_path ="/home/vinicios/reps/hcac/datasets/ebrasui.data"
    dataset = load_dataset_from_file(input_path)
    sentences,labels = load_dataset(dataset)
    # sentences = preprocessing(sentences)
    _max_dim = (0,0)
    embeddings, max_dim = get_embeddings(sentences, max_dim=_max_dim)
    #dense_dataset = do_padding(embeddings, max_dim, labels)
    embeddings = np.array(embeddings)
    print("embeddings shape",embeddings.shape)
    label = np.array(labels).reshape(embeddings.shape[0],1)
    dense_dataset = np.append(embeddings,label, axis=1)
    return dense_dataset

def save_to_file(dense_dataset):
    output_path = "/home/vinicios/reps/hcac/datasets/ebrasui.data"
    file_handler = open(output_path, 'wb')
    pickle.dump(dense_dataset, file_handler)
    # np.savetxt(output, dense_dataset)

if __name__ == '__main__':
    embeddings = []
    max_dim = (0,0)
    # dense_dataset = np.array(from_text_to_embeddings())
    dense_dataset = from_text_to_embeddings()
    save_to_file(dense_dataset)
