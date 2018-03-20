import os,sys
import re
import numpy as np
from collections import Counter
from nltk.corpus import stopwords 
from nltk.stem.snowball import SnowballStemmer
from mpmath import mp
from nltk.stem import PorterStemmer
import math

"""Stemmer object"""
stemmer = SnowballStemmer('english')

def get_data(path): 
    """ This function gets all the contents of 
    the file and concatenates them into word_list
    """
    word_list=""
    dirs = os.listdir(path)
    for i in range(len(dirs)):
        with open(path+dirs[i],"r",encoding="utf-8") as f:
                file = f.read()
                file=file.lower()
                word_list+=file+" "
                f.close()
    print(word_list)
    return word_list

def process_data(files):
    """This function further processes the 
    html tags, take out the words only and
     then eliminates the stop words and stems 
    the words using the nltk library
    """
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr,'',files)
    word=re.findall(r'[a-z]+',cleantext)
    stops = set(stopwords.words("english")) 
    filtered_words = [w for w in word if not w in stops] 
    stemmed_words=[stemmer.stem(w) for w in filtered_words]   
    return stemmed_words

def probability_calculation(word_counter,total_word,total_vocab):
    """Takes the word_counter object which is
    made from the Counter.Replaces the individual 
    count of the each words with the conditional 
    probability for each word.
    """
    for k,v in word_counter.items():
        word_counter[k]=(v+1)/(total_word+total_vocab)
    return word_counter

def test(path):
    """This function takes the test files and multiplies
    all of the probabilities considered as positive and 
    considered as negative.It then compares them and 
    classifies them.
    """
    dirs = os.listdir(path)
    result = 0
    data_count = 0
    for i in range(len(dirs)):
        with open(path+dirs[i],"r",encoding="utf-8") as f:
                file = f.read()
                test_data_pos = file.lower()
                f.close()
                stemmed_test_pos = process_data(test_data_pos)
                pos_class_probability = 1
                neg_class_probability = 1
                data_count = data_count + 1
                for word in stemmed_test_pos:                   
                    pos_class_probability = mp.fmul(pos_class_probability,pos_probability.get(word,1))# multiplying all the positive probability of the words in the test file 
                    neg_class_probability = mp.fmul(neg_class_probability,neg_probability.get(word,1)) # multiplying all the negative probablity of the words in the test file                          
                if (pos_class_probability > neg_class_probability): 
                    result = result + 1 #result stores the number of positive result obtained
    return result, data_count
                

"""Training for the positive data"""
path_pos="D://bayes//train//pos//"
pos_data = get_data(path_pos)
pos_stemmed_words = process_data(pos_data)
pos_counter = Counter(pos_stemmed_words)
pos_vocab_length=len(pos_counter)
pos_totalword_length=len(pos_stemmed_words)
pos_probability=probability_calculation(pos_counter,pos_vocab_length,pos_totalword_length)




"""Training for the negative data"""
path_neg="D://bayes//train//neg//"
neg_data = get_data(path_neg)
neg_stemmed_words = process_data(neg_data)
neg_counter = Counter(neg_stemmed_words)
neg_vocab_length=len(neg_counter)
neg_totalword_length=len(neg_stemmed_words)

neg_probability=probability_calculation(neg_counter,neg_vocab_length,neg_totalword_length)

"""Testing"""
actual_positive = 0
actual_negative = 0
predicted_positive = 0
predicted_negative = 0

"""Testing for the positive data"""
path_pos_test = "D://bayes//test//pos//" #path for test positive
test_positive = test(path_pos_test) #gets the number of positive prediction
predicted_positive = test_positive[0]
actual_positive = test_positive[1]

"""Testing for the negative data"""
path_neg_test = "D://bayes//test//neg//"#path for test negative
test_negative = test(path_neg_test) 
predicted_negative = test_negative[1] - test_negative[0] 
actual_negative = test_negative[1]
        
print(actual_positive)
print(actual_negative)
print(predicted_positive)
print(predicted_negative)

"""Calculation of the accuracy"""
total_accuracy=(predicted_positive+predicted_negative)/(actual_positive+actual_negative)
print(total_accuracy)