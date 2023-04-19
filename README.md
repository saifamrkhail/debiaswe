# Debiasing Word Embeddings

This repository contains code for debiasing word embeddings based on the paper "Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings" by Bolukbasi et al.

## Getting Started

To use the code in this repository, you will need to have Python 3 installed, along with the following packages:

- numpy
- scipy
- scikit-learn
- matplotlib
- gensim

You can install these packages using pip:

```bash
$ pip install numpy scipy scikit-learn matplotlib gensim
```

## Usage

The main file in this repository is debias.py, which contains code for loading word embeddings, debiasing them, and evaluating the debiased embeddings on various tasks.

To use this code, simply run the debias.py file with the appropriate command-line arguments. For example, to load word embeddings from a file and debias them using the "hard" debiasing method, you can run:

```bash
python debias.py --embedding_file embeddings.txt --debias_method hard
```

This will output the debiased embeddings to a file called embeddings_debiased.txt.

## Demo

In the `demo.py` you will find a demonstration of how to use the code. The demo consists of:

- Loading and visualizing word embeddings
- Debiasing word embeddings using various methods
- Evaluating the debiased embeddings on gender-related tasks


## Python scripts:
- **learn_gender_specific.py**: given a word embedding and a seed set of gender-specific words (like <i>king</i>, <i>she</i>, etc.), it learns a much larger list of gender-specific words
- **debias.py**: given a word embedding, sets of gender-pairs, gender-specific words, and pairs to equalize, it outputs a new word embedding. This version basically reads/writes word2vec binary file format.  

```
python learn_gender_specific.py ../embeddings/GoogleNews-vectors-negative300.bin 50000 ../data/gender_specific_seed.json gender_specific_full.json
```

```
python debias.py ../embeddings/GoogleNews-vectors-negative300.bin ../data/definitional_pairs.json ../data/gender_specific_full.json ../data/equalize_pairs.json ../embeddings/GoogleNews-vectors-negative300-hard-debiased.bin
```


We also have seed data used to debias and crowd data used to evaluate the embeddings.

## Data files:
- **gender_specific_seed.json**: A list of 218 gender-specific words
- **gender_specific_full.json**: A list of 1441 gender-specific words
- **definitional_pairs.json**: The ten pairs of words we use to define the gender direction
- **equalize_pairs.json**: Some crowdsourced F-M pairs of words that represent gender direction


All external files can be found on the Googel Drive of the authors of the paper. [Google Drive](https://drive.google.com/drive/folders/0B5vZVlu2WoS5dkRFY19YUXVIU2M?resourcekey=0-rZ1HR4Fb0XCi4HFUERGhRA&usp=sharing).
