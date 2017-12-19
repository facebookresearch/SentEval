# SentEval

SentEval is a library for evaluating the quality of sentence embeddings. We assess their generalization power by using them as features on a broad and diverse set of "transfer" tasks (more details [here](https://arxiv.org/abs/1705.02364)). Our goal is to ease the study and the development of general-purpose fixed-size sentence representations.

## Dependencies

This code is written in python. The dependencies are:

* Python 2.7 (with recent versions of [NumPy](http://www.numpy.org/)/[SciPy](http://www.scipy.org/))
* [Pytorch](http://pytorch.org/) >= 0.2 (**recent new pytorch version**)
* [scikit-learn](http://scikit-learn.org/stable/index.html)>=0.18.0

## Tasks

SentEval allows you to evaluate your sentence embeddings as features for the following tasks:
* Binary classification: [MR](https://nlp.stanford.edu/~sidaw/home/projects:nbsvm) (movie review), [CR](https://nlp.stanford.edu/~sidaw/home/projects:nbsvm) (product review), [SUBJ](https://nlp.stanford.edu/~sidaw/home/projects:nbsvm) (subjectivity status), [MPQA](https://nlp.stanford.edu/~sidaw/home/projects:nbsvm) (opinion-polarity), [SST](https://nlp.stanford.edu/sentiment/index.html) (Stanford sentiment analysis)
* Multi-class classification: [TREC](http://cogcomp.cs.illinois.edu/Data/QA/QC/) (question-type classification), [SST](http://www.aclweb.org/anthology/P13-1045) (fine-grained Stanford sentiment analysis)
* Entailment (NLI): [SNLI](https://nlp.stanford.edu/projects/snli/) (caption-based NLI), [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/) (Multi-genre NLI), [SICK](http://clic.cimec.unitn.it/composes/sick.html) (Sentences Involving Compositional Knowledge, entailment)
* Semantic Textual Similarity: [STS12](https://www.cs.york.ac.uk/semeval-2012/task6/), [STS13](http://ixa2.si.ehu.es/sts/) (-SMT), [STS14](http://alt.qcri.org/semeval2014/task10/), [STS15](http://alt.qcri.org/semeval2015/task2/), [STS16](http://alt.qcri.org/semeval2016/task1/)
* Semantic Relatedness: [STSBenchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark#Results), [SICK](http://clic.cimec.unitn.it/composes/sick.html)
* Paraphrase detection: [MRPC](https://aclweb.org/aclwiki/Paraphrase_Identification_(State_of_the_art)) (Microsoft Research Paraphrase Corpus)
* Caption-Image retrieval: [COCO](http://mscoco.org/) dataset (with ResNet-101 2048d image embeddings)

[more details on the tasks](https://arxiv.org/pdf/1705.02364.pdf)

## Download datasets
To get all the transfer tasks datasets, run (in data/):
```bash
./get_transfer_data_ptb.bash
```
This will automatically download and preprocess the datasets, and put them in data/senteval_data (warning: for MacOS users, you may have to use p7zip instead of unzip).

WARNING: Extracting the [MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398) MSI file requires the "[cabextract](https://www.cabextract.org.uk/#install)" command line (i.e *apt-get/yum install cabextract*).

## Example (average word2vec) : examples/bow.py

### examples/bow.py

In examples/bow.py, we evaluate the quality of the average(GloVe) embeddings.

To get GloVe embeddings [2GB], run (in examples/):
```bash
./get_glove.bash
```

To reproduce the results for avg(GloVe) vectors, run (in examples/):  
```bash
python bow.py
```

As required by SentEval, this script implements two functions: **prepare** (optional) and **batcher** (required) that turn text sentences into sentence embeddings. Then SentEval takes care of the evaluation on the transfer tasks using the embeddings as features.

### examples/infersent.py

To get the **[InferSent](https://www.github.com/facebookresearch/InferSent)** model and reproduce our results, download our best models and run infersent.py (in examples/):
```bash
curl -Lo examples/infersent.allnli.pickle https://s3.amazonaws.com/senteval/infersent/infersent.allnli.pickle
curl -Lo examples/infersent.snli.pickle https://s3.amazonaws.com/senteval/infersent/infersent.snli.pickle
```

## How SentEval works

To evaluate your own sentence embedding method, you will need to implement two functions: 

1. **prepare** (sees the whole dataset of each task and can thus construct the word vocabulary, the dictionary of word vectors etc)
2. **batcher** (transforms a batch of text sentences into sentence embeddings)


### 1.) prepare(params, samples) (optional)

batcher only sees one batch at a time while the *samples* argument of *prepare* contains all the sentences of a task.

```
prepare(params, samples)
```
* *batch*: numpy array of text sentences
* *params*: senteval parameters (note that "prepare" outputs are stored in params).
* *output*: None. Any "output" computed in this function is stored in "params" and can be further used by *batcher*.

*Example*: in bow.py, prepare is is used to build the vocabulary of words and construct the "params.word_vect* dictionary of word vectors.


### 2.) batcher(params, batch)
```
batcher(params, batch)
```
* *batch*: numpy array of text sentences (of size params.batch_size)
* *params*: senteval parameters (note that "prepare" outputs are stored in params).
* *output*: numpy array of sentence embeddings (of size params.batch_size)

*Example*: in bow.py, batcher is used to compute the mean of the word vectors for each sentence in the batch using params.word_vec. Use your own encoder in that function to encode sentences.


### 3.) evaluation on transfer tasks

After having implemented the batch and prepare function for your own sentence encoder,

1) to perform the actual evaluation, first import senteval and define a SentEval object:
```python
import senteval
se = senteval.SentEval(params, batcher, prepare)
```
(to import senteval, you can either add senteval path to your pythonpath, use sys.path.insert or "*pip install git+https://github.com/facebookresearch/SentEval*")

2) define the set of transfer tasks on which you want SentEval to perform evaluation and run the evaluation: 
```python
transfer_tasks = ['MR', 'SICKEntailment', 'STS14', 'STSBenchmark']
results = se.eval(transfer_tasks)
```
The current list of available tasks is:
```python
['CR', 'MR', 'MPQA', 'SUBJ', 'SST', 'TREC', 'MRPC', 'SNLI',
'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval',
'STS12', 'STS13', 'STS14', 'STS15', 'STS16']
```
Note that the tasks of image-caption retrieval, SICKRelatedness, STSBenchmark and SNLI require pytorch and the use of a GPU. For the other tasks, 
setting *usepytorch* to False will make them run on the CPU (with sklearn), which can be faster for small embeddings but slower for large embeddings.
## SentEval parameters
SentEval has several parameters (only task_path is required):
* **task_path** (str): path to data, generated by data/get_transfer_data.py
* **seed** (int): random seed for reproducability (default: 1111)
* **usepytorch** (bool): use pytorch or scikit learn (when possible) for logistic regression (default: True). Note that sklearn is quite fast for small dimensions. Use pytorch for SNLI.
* **classifier** (str): if usepytorch, choose between 'LogReg' and 'MLP' (tanh) (default: 'LogReg')
* **nhid** (int): if usepytorch and classifier=='MLP' choose nb hidden units (default: 0)
* **batch_size** (int): size of minibatch of text sentences provided to "batcher" (sentences are sorted by length). Note that this is not the batch_size used by pytorch logistic regression, which is fixed.
* **kfold** (int): k in the kfold-validation. Set to 10 to be comparable to published results (default: 5)
* ... and any parameter you want to have access to in "batcher" or "prepare" functions.


## References

Please cite [1](https://arxiv.org/abs/1705.02364) [2](https://arxiv.org/abs/1707.06320) if using this code for evaluating sentence embedding methods.

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

### Learning Visually Grounded Sentence Representations
 
[2] D. Kiela, A. Conneau, A. Jabri, M. Nickel, [*Learning Visually Grounded Sentence Representations*](https://arxiv.org/abs/1707.06320)

```
@article{kiela2017learning,
  title={Learning Visually Grounded Sentence Representations},
  author={Kiela, Douwe and Conneau, Alexis and Jabri, Allan and Nickel, Maximilian},
  journal={arXiv preprint arXiv:1707.06320},
  year={2017}
}
```

## License
SentEval is BSD-licensed.

Contact: [aconneau@fb.com](mailto:aconneau@fb.com), [dkiela@fb.com](mailto:dkiela@fb.com)
