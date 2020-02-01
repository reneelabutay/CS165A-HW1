import numpy as np
import re
import operator
from collections import Counter

def generate_word_collection(file_name):

  f = open(file_name, encoding="ISO-8859-1")
  text_lines = f.readlines() #each line is an element in the array

  whole_collection = Counter()

  for line in text_lines:
    arr = line.split(' ')
    arr = arr[:-1]
    for word in arr:
      whole_collection[word] += 1

  return whole_collection

word_collection = generate_word_collection("training.txt")

def print_top_k(word_collection, k):
  top = word_collection.most_common(k)
  for i in top:
    print(i)

print_top_k(word_collection, 20)

"""
Output example:
('the', 190806)
('of', 116447)
('and', 102422)
('to', 89251)
('a', 72558)
('in', 56028)
('i', 45308)
('that', 39378)
('he', 38160)
('it', 34361)
('was', 33834)
('his', 28771)
('Ã¢', 28082)
('with', 25176)
('as', 25032)
('for', 23788)
('is', 22857)
('you', 22808)
('her', 21188)
('had', 20869)
"""