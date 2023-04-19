"""
This is a demonstration of gender bias in NLP
by using  debiaswe and word2vec word embeddings.

author: Manuel KRANZL
email: 	ai22m038@technikum-wien.at
author: Saifur Rahman RAHMANI
email: ai22m055@technikum-wien.at
"""

from __future__ import print_function, division 
import matplotlib.pyplot as plt
import pprint
import json
import random
import numpy as np

import debiaswe as dwe
import debiaswe.we as we
from debiaswe.we import WordEmbedding
from debiaswe.data import load_professions

#First Demo: Gender Bias in Word Embeddings

#Step 1: Gender bias in word embeddings
#
#We use word2vec word embeddings trained on Google News.
#word2vec contains 300-dimensional vectors for 3 million words and phrases.
#To make the demo run faster, we use a smaller version of word2vec.
#We downloaded w2v_gnews_small.txt, which is in embeddings/ directory
 
# load google news word2vec
print('\nLoading Google News Word2Vec')
E = WordEmbedding('./embeddings/w2v_gnews_small.txt')

# load professions
professions = load_professions()
profession_words = [p[0] for p in professions]


#Step 2: Define gender direction
#
#We define gender direction by the direciton of she - he
#because they are frequent and do not have fewer alternative word senses
#(e.g., man can also refer to mankind). 
#In the paper, we discuss alternative approach for defining gender direction (e.g., using PCA).

# gender direction
print('\nGender Direction')
v_gender = E.diff('she', 'he')
print('\n')

#Step 3: Generating analogies
#
#An example for an analogy would be,
#“man is to king as woman is to x” (denoted as man:king :: woman:x),
#simple arithmetic of the embedding vectors finds that x=queen is the best answer.
#
#Similarly, x=Japan is returned for Paris:France :: Tokyo:x.
#We show that the word embedding model generates gender-streotypical analogy pairs.
#These word pairs should be well aligned with gender direction as well as within a short distance
#from each other to preserve topic consistency. 

# analogies gender
print('\nGender Bias in Analogies')
a_gender = E.best_analogies_dist_thresh(v_gender)
for (a,b,c) in a_gender:
    print(a+"-"+b)


#Step 4: Analyzing gender bias in word vectors asscoiated with professions
#
#Next, we show that many occupations are unintendedly associated with either male of female
#by projecting their word vectors onto the gender dimension. 
#The script will output the profession words sorted with respect to the projection score in the direction of gender.


#profession analysis gender
#definitional female -1.0 -> definitional male 1.0
#stereotypical female -1.0 -> stereotypical male 1.0
print('\nAnalysing gender bias in professions')
print('Gender Bias Scores of Profession Words')
print('definitional female -1.0 -> definitional male 1.0')
print('stereotypical female -1.0 -> stereotypical male 1.0')
sp = sorted([(E.v(w).dot(v_gender), w) for w in profession_words])
pp = pprint.PrettyPrinter()
pp.pprint(sp[0:20])
pp.pprint(sp[-20:])

scores = [score for score, _ in sp]
words = [word for _, word in sp]

# Create a bar plot of the scores
fig, ax = plt.subplots()
ax.bar(range(len(words)), scores)
ax.set_xticks(range(len(words)))
ax.set_xticklabels(words, rotation=90, fontsize=8)
ax.set_xlabel('Words')
ax.set_ylabel('Gender Bias Score')
ax.set_title('Gender Bias Scores of Profession Words')
#unfortunately, the word on the x-axis are not readable, but look at output of console 
#to see the words with its score

#Step 5: Debiasing word embeddings
print('\nDebiasing Word Embeddings')
from debiaswe.debias import debias

# Load gender related word lists
print('Load gender related word lists')
with open('./data/definitional_pairs.json', "r") as f:
    defs = json.load(f)
print("definitional", defs)

with open('./data/equalize_pairs.json', "r") as f:
    equalize_pairs = json.load(f)

with open('./data/gender_specific_seed.json', "r") as f:
    gender_specific_words = json.load(f)
print("gender specific", len(gender_specific_words), gender_specific_words[:10])

#do debiasing
print('\nDo Debiasing')
debias(E, gender_specific_words, defs, equalize_pairs)

# profession analysis gender
print('\nAnalysing gender bias in professions after debiasing')
sp_debiased = sorted([(E.v(w).dot(v_gender), w) for w in profession_words])
pp.pprint(sp_debiased[0:20])
pp.pprint(sp_debiased[-20:])

scores_debiased = [score for score, _ in sp_debiased]
words_debiased = [word for _, word in sp_debiased]



# Create a bar plot of the scores
fig, ax = plt.subplots()
ax.bar(range(len(words_debiased)), scores_debiased)
ax.set_xticks(range(len(words_debiased)))
ax.set_xticklabels(words_debiased, rotation=90, fontsize=8)
ax.set_xlabel('Words after Debiasing')
ax.set_ylabel('Gender Bias Score after Debiasing')
ax.set_title('Gender Bias Scores of Profession Words after Debiasing')
#unfortunately, the word on the x-axis are not readable, but look at output of console 
#to see the words with its score

# analogies gender
print('\nGender Bias in Analogies after debiasing')
a_gender_debiased = E.best_analogies_dist_thresh(v_gender)
for (a,b,c) in a_gender_debiased:
    print(a+"-"+b)

#plot in end to allow complete printing to console
plt.show()
