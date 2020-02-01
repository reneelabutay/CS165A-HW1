import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
# from sklearn.linear_model import LogisticRegression
from math import log
import re
import operator

def generate_word_collection(file_name):
    '''
    @input:
      file_name: a string. should be either "training.txt" or "texting.txt"
    @return:
      word_collection: a list of all words in the given file, sorted in the alphabetical order.
            Note1: This is slightly different from the interface in Question 2(c).
            Note2: Assuming you have a word collection ['c', 'ab', 'd'], your output should be a sorted list as ['ab', 'c', 'd'].
    '''
    f = open(file_name, encoding='ISO-8859-1')
    # Step1: Generate word_collection

    text_lines = f.readlines()

    word_collection = []

    for line in text_lines:
        arr = line.split(' ')
        arr = arr[:-1]
        for word in arr:
            if word not in word_collection:
                word_collection.append(word)

    # Step2: Sort your word_collection alphabetically. - DONE
    word_collection = sorted(word_collection)

    return word_collection

# Note that len(word_collection) == 10001. For varifying your code. BUT I GOT 10000 YIKES
word_collection = generate_word_collection("training.txt")

'''
Sample result:
['a', 'aa', 'aad', 'ab', 'abandon', 'abandoned', 'abashed', 'abbey', 'abide', 'abiding', ...]

Note: No output are required. Just for varifying your code. Only first few words are printed for reading convenience.
'''


# Question 2(a)
def bag_of_word_encoding(sentence, word_collection):
    """
    @input:
      sentence: a string. Stands for "D" in the problem description.
                One example is "wish for solitude he was twenty years of age ".
      word_collection: a list. Refer to the output of generate_word_collection(file_name).
    @output:
      encoded_array: a sparse vector based on library scipy.sparse, csr_matrix.
    """
    word_list = list(sentence.split(" "))
    word_list.pop()

    dictionary = dict.fromkeys(word_collection, 0)

    # for-loop adds frequencies
    for word in word_list:
        count = dictionary.get(word, 0)
        dictionary[word] = count + 1.0

    list_arr = list(dictionary.values())

    encoded_array = csr_matrix(list_arr)
    return encoded_array

sentence = "wish for solitude he was twenty years of age "
print(bag_of_word_encoding(sentence, word_collection))
'''
Sample output:
  (0, 223)	1.0
  (0, 3497)	1.0
  (0, 4086)	1.0
  (0, 5975)	1.0
  (0, 8141)	1.0
  (0, 9234)	1.0
  (0, 9623)	1.0
  (0, 9811)	1.0
  (0, 9963)	1.0
'''


# Question 2(b)
def N_Gram(sentence, N):
    tokens = sentence.split(" ")
    tokens.pop()
    sequences = [tokens[i:] for i in range(N)]
    ngrams = zip(*sequences)
    return [" ".join(ngram) for ngram in ngrams]

sentence = "wish for solitude he was twenty years of age "
print(N_Gram(sentence, 3))

'''
Sample output:
['wish for solitude', 'for solitude he', 'solitude he was', 'he was twenty', 'was twenty years', 'twenty years of']
'''

# Question 2(c)
def get_TF(term, document):
    '''
    @input:
        term: str. a word (e.g., cat, dog, fish, are, happy)
        document: str. a sentence (e.g., "wish for solitude he was twenty years of age ").
    @output:
        TF: float. frequency of term in the document.
    '''
    document = document.strip()
    document = document.split(" ") #list of each word in document

    total_words = len(document)
    frequency = 0
    for word in document:
        if word == term:
            frequency += 1

    TF = frequency/total_words
    return TF

def get_IDF(term, file_name):
    '''
    @input:
      term: str. a word (e.g., cat, dog, fish, are, happy)
      file_name: a string. should be either "training.txt" or "texting.txt"
    @output:
      IDF: float. IDF = log_e(Total number of documents / Nmber of documents with term t in it)
    '''

    f = open(file_name, encoding='ISO-8859-1')
    all_documents = f.readlines() #array of each document (line)
    num_terms = 0

    for document in all_documents:
        list_sentences = document.split(' ')
        for word in list_sentences:
            if word == term:
                num_terms += 1
                break

    if (num_terms == 0):
        num_terms += 1

    IDF = (len(all_documents)) / (num_terms)
    return np.log(IDF)

def get_TF_IDF(term, document, filename):
    '''
    @input:
      term: str. a word (e.g., cat, dog, fish, are, happy)
      document: str. a sentence (e.g., "wish for solitude he was twenty years of age ").
      file_name: a string. should be either "training.txt" or "texting.txt"
    @output:
      TF_IDF: float. Equal to TF*IDF.
    '''
    TF = get_TF(term, document)
    IDF = get_IDF(term, filename)
    TF_IDF = TF * IDF
    return TF_IDF


def TF_IDF_encoding(word_collection, filename, document):
    '''
    @input:
      word_collection: a list. Refer to the output of generate_word_collection(file_name).
      file_name: a string. should be either "training.txt" or "texting.txt"
      document: str. a sentence (e.g., "wish for solitude he was twenty years of age ").
    @output:
      encoded_array: a sparse vector based on library scipy.sparse, csr_matrix. Contain the TF_IDF_encoding of a given document
                     (or a single line in the training.txt or testing.txt).
    '''
    document.strip()
    doc_list = list(document.split(" "))  #list of words from the sentence

    dictionary = dict.fromkeys(word_collection, 0) #dictionary of word_collection, all empty values

    # for-loop adds TD-IDF values
    for word in doc_list:
        if(dictionary.get(word) != None):
            dictionary[word] = get_TF_IDF(word, sentence, filename)

    list_arr = list(dictionary.values())
    encoded_array = csr_matrix(list_arr)
    return encoded_array

sentence = "wish for solitude he was twenty years of age "
print(TF_IDF_encoding(word_collection, "training.txt", sentence))

'''
Sample output:
  (0, 223)	0.20907601963956024
  (0, 3497)	0.0002595621997275229
  (0, 4086)	0.008943027879187394
  (0, 5975)	0.00011116670373151355
  (0, 8141)	0.3972834187563259
  (0, 9234)	0.2057232748482032
  (0, 9623)	0.006560282292642595
  (0, 9811)	0.18511202932472162
  (0, 9963)	0.11087359839644073
'''