#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from utils import split_data_target
import string
import sys

def preprocessing(data):
    #this function removes all the punctuation in the text and set all characters to lowercase
    a,b = data.shape
    for i in range(a):
        data[i][0] = data[i][0].lower()
        data[i][0] = data[i][0].translate(string.punctuation)
    return np.array(data)

def text_to_bow():
    #this function will get the text file, makes the preprocessing and return a numpy array with the bag of words
    divisor = 1
    if (len(sys.argv) < 2):
        print("ERROR: No input file! \n Usage: python3 create_bow_dataset_from_text.py <text dataset file> ")
        exit(1)
    input = sys.argv[1]
    data = []
    target = []
    file = open(input, "r")
    for line in file:
        # print(line)
        # data.append([str(line.split(";")[0]).decode('utf8')])
        data.append([str(line.split(";")[0])])
        target.append([int(line.split(";")[1])])
    file.close()
    data = np.array(data)
    data = preprocessing(data)
    target = np.array(target)
    # target = target.reshape(target.shape[1], target.shape[0])


    filtered_sentence = []
    fs = []
    word_tokens = []
    stop_words = set(stopwords.words("portuguese"))
    stemmer = SnowballStemmer("portuguese")

    size = data.shape[0]/divisor
    cont = 0
    for i in range(int(data.shape[0]/divisor)):
        cont +=1
        word_tokens.append(word_tokenize(data[i][0]))
        fs = [w for w in word_tokens[i] if not w in stop_words]
        if len(fs)>0:
            for i in range(len(fs)):
                filtered_sentence.append(stemmer.stem(fs[i]))

    filtered_sentence = list(set(filtered_sentence))
    bow = []
    number_of_words = len(filtered_sentence)
    cont = 0
    for i in range(data.shape[0]):
        cont +=1
        v = np.zeros(number_of_words)
        for word in word_tokens[i]:
            if (word in filtered_sentence):
                v[filtered_sentence.index(word)] +=1
        bow.append(v)
    bow = np.array(bow)
    # print(bow.shape, target.shape)
    data = np.append(bow, target[0:target.shape[0]], axis=1)
    return data

def save_dataset(data):
    output = sys.argv[2]
    np.savetxt(output, data)

if __name__ == '__main__':
    data = text_to_bow()
    save_dataset(data)
    #to execute this program, use:
    # $python preprocessing <input textual data file path> <output numpy matrix file path>
