# SentEval

SentEval is a library for evaluating the quality of sentence embeddings as features for a broad and diverse set of "transfer" tasks. It is aimed to ease the study and the development of general-purpose fixed-size sentence representations.

## Dependencies

This code is written in python. The dependencies are :

* Python 2.7 (with recent versions of [NumPy](http://www.numpy.org/)/[SciPy](http://www.scipy.org/))
* [Pytorch](http://pytorch.org/) >= 0.12
* [scikit-learn](http://scikit-learn.org/stable/index.html)>=0.18.0


## Tasks

SentEval allows you to evaluate your sentence embeddings as features for the following tasks :
* Binary classification : [MR](https://nlp.stanford.edu/~sidaw/home/projects:nbsvm) (movie review), [CR](https://nlp.stanford.edu/~sidaw/home/projects:nbsvm) (product review), [SUBJ](https://nlp.stanford.edu/~sidaw/home/projects:nbsvm) (subjectivity status), [MPQA](https://nlp.stanford.edu/~sidaw/home/projects:nbsvm) (opinion-polarity), [SST](https://nlp.stanford.edu/sentiment/index.html) (Stanford sentiment analysis)
* Multi-class classification : [TREC](http://cogcomp.cs.illinois.edu/Data/QA/QC/) (question-type classification), [SST](http://www.aclweb.org/anthology/P13-1045) (fine-grained Stanford sentiment analysis)
* Entailment (NLI) : [SNLI](https://nlp.stanford.edu/projects/snli/) (caption-based NLI), [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/) (Multi-genre NLI), [SICK](http://clic.cimec.unitn.it/composes/sick.html) (Sentences Involving Compositional Knowledge, entailment)
* Semantic Textual Similarity : [STSBenchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark#Results), [STS14](http://alt.qcri.org/semeval2014/task10/), [SICK](http://clic.cimec.unitn.it/composes/sick.html) (relatedness)
* Paraphrase detection : [MRPC](https://aclweb.org/aclwiki/index.php?title=Paraphrase_Identification_(State_of_the_art)) (Microsoft Research Paraphrase Corpus)
* Caption-Image retrieval : [COCO](http://mscoco.org/) dataset (with ResNet-101 2048d image embeddings)

[more details on the tasks](https://arxiv.org/pdf/1705.02364.pdf)

## Download datasets
To get all the transfer tasks datasets, run (in data/) :
```bash
./get_transfer_data.bash
```
This will automatically download and preprocess the datasets, and put them in data/senteval_data.

WARNING : Extracting the [MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398) MSI file requires the "[cabextract](https://www.cabextract.org.uk/#install)" command line (i.e *apt-get/yum install cabextract*).

## Example (average word2vec) : examples/bow.py

### examples/bow.py

In examples/bow.py, we evaluate the quality of the average(GloVe) embeddings.

To get GloVe embeddings, run (in examples/):
```bash
./get_glove.bash
```

To reproduce the results for avg(GloVe) vectors, run (in examples/):  
```bash
python bow.py
```

As required by SentEval, this script implements two functions : **batcher** (required), and **prepare** (optional) that turn text sentences into sentence embeddings. Then SentEval takes care of the evaluation on the transfer tasks using the embeddings as features.

## How SentEval works

To evaluate your own sentence embedding method, you will need to implement two functions : 

1. **prepare** (sees the whole dataset of each task and can construct the word vocabulary, the dictionary of word vectors etc)
2. **batcher** (transforms a minibatch of text sentences into sentence embeddings)


### 1.) prepare(params, samples) (optional)

batcher only sees one batch at a time while the *samples* argument of *prepare* contains all the sentences of a task.

```
prepare(params, samples)
```
* *batch* : numpy array of text sentences
* *params* : senteval parameters (note that "prepare" outputs are stored in params).
* *output* : None. Any "output" computed in this function is stored in "params" and can be further used by *batcher*.

*Example* : in bow.py, prepare is is used to build the vocabulary of words and construct the "params.word_vect* dictionary of word vectors.


### 2.) batcher(batch, params)
```
batcher(batch, params)
```
* *batch* : numpy array of text sentences (of size params.batch_size)
* *params* : senteval parameters (note that "prepare" outputs are stored in params).
* *output* : numpy array of sentence embeddings (of size params.batch_size)

*Example* : in bow.py, batcher is used to compute the mean of the word vectors for each sentence in the batch using params.word_vec. Use your own encoder in that function to encode sentences.


### 3.) evaluation on transfer tasks

After having implemented the batch and prepare function for your own sentence embedding model,

1) to perform the actual evaluation, first import senteval and define a SentEval object :
```python
import senteval
se = senteval.SentEval(batcher, prepare, params_senteval)
```
(to import senteval, you can either add senteval path to your pythonpath or use sys.path.insert)

2) define the set of transfer tasks on which you want SentEval to perform evaluation and run the evaluation : 
```python
transfer_tasks = ['SST', 'SNLI']
results = se.eval(transfer_tasks)
```
The current list of available tasks is :
```python
['CR', 'MR', 'MPQA', 'SUBJ', 'SST', 'TREC', 'MRPC', 'SNLI', 'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'STS14', 'ImageAnnotation']
```

## SentEval parameters
SentEval has several parameters (all have default settings except task_path):
* **task_path** (str) : path to data, generated by data/get_transfer_data.py
* **seed** (int) : random seed for reproducability (default : 1111)
* **usepytorch** (bool) : use pytorch or scikit learn for logistic regression (default : True)
* **classifier** (str) : if usepytorch, choose between 'LogReg' and 'MLP' (tanh) (default : 'LogReg')
* **nhid** (int) : if usepytorch and classifier=='MLP' choose nb hidden units (default : 0)
* **batch_size** (int) : size of minibatch of text sentences provided to "batcher" (sentences are sorted by length) (note that this is not the batch_size used by pytorch logistic regression, which is fixed)
* **verbose** (int) : 2->debug, 1->info, 0->warning (default : 2)
* ... and any parameter you want to have access to in "batcher" or "prepare" functions.


## References

Please cite [1](https://arxiv.org/abs/1705.02364) if using this code for evaluating sentence embedding methods.

### Supervised Learning of Universal Sentence Representations from Natural Language Inference Data

[1] A. Conneau, D. Kiela, H. Schwenk, L. Barrault, A. Bordes, [*Supervised Learning of Universal Sentence Representations from Natural Language Inference Data*](https://arxiv.org/abs/1705.02364)

```
@article{conneau2017supervised,
  title={Supervised Learning of Universal Sentence Representations from Natural Language Inference Data},
  author={Conneau, Alexis and Kiela, Douwe and Schwenk, Holger and Barrault, Loic and Bordes, Antoine},
  journal={arXiv preprint arXiv:1705.02364},
  year={2017}
}
```
