# SentEval

SentEval is a library for evaluating the quality of sentence embeddings as features for a broad and diverse set of "transfer" tasks. It is aimed to ease the study and the development of better general-purpose fixed-size sentence representations (see [InferSent](https://arxiv.org/pdf/1705.02364.pdf) for a comparison).

## Dependencies

This code is written in python. The dependencies are :

* Python 2.7 (with recent versions of [NumPy](http://www.numpy.org/)/[SciPy](http://www.scipy.org/))
* [Pytorch](http://pytorch.org/) >= 0.12
* [scikit-learn](http://scikit-learn.org/stable/index.html)>="0.18.0"


## Tasks

See [here](https://arxiv.org/pdf/1705.02364.pdf) for a detailed description of the available tasks.
* Binary classification : [MR](https://nlp.stanford.edu/~sidaw/home/projects:nbsvm) (movie review), [CR](https://nlp.stanford.edu/~sidaw/home/projects:nbsvm) (product review), [SUBJ](https://nlp.stanford.edu/~sidaw/home/projects:nbsvm) (subjectivity status), [MPQA](https://nlp.stanford.edu/~sidaw/home/projects:nbsvm) (opinion-polarity), [SST](https://nlp.stanford.edu/sentiment/index.html) (Stanford sentiment analysis)
* Multi-class classification : [TREC](http://cogcomp.cs.illinois.edu/Data/QA/QC/) (question-type classification), [SST](http://www.aclweb.org/anthology/P13-1045) (fine-grained Stanford sentiment analysis)
* Entailment (NLI) : [SNLI](https://nlp.stanford.edu/projects/snli/) (caption-based NLI), [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/) (Multi-genre NLI), [SICK](http://clic.cimec.unitn.it/composes/sick.html) (Sentences Involving Compositional Knowledge, entailment)
* Semantic Textual Similarity : [STSBenchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark#Results), [STS14](http://alt.qcri.org/semeval2014/task10/), [SICK](http://clic.cimec.unitn.it/composes/sick.html) (relatedness)
* Paraphrase detection : [MRPC](https://aclweb.org/aclwiki/index.php?title=Paraphrase_Identification_(State_of_the_art)) (Microsoft Research Paraphrase Corpus)
* Caption-Image retrieval : [COCO](http://mscoco.org/) dataset (with ResNet-101 2048d image embeddings)


## Download datasets
To get all the transfer tasks datasets, run (in data/) :
```bash
./get_transfer_data.bash
```
This will automatically download and preprocess the datasets, and put them in data/senteval_data.

WARNING : Downloading the [MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398) dataset requires the "[cabextract](https://www.cabextract.org.uk/#install)" command line (sudo apt-get/yum/brew install cabextract) to extract the provided Microsoft-specific MSI file.

## Example (average word2vec) : examples/bow.py

### examples/bow.py

In examples/bow.py, we provide a minimal example for evaluating the quality of the *average of word vectors* (word2vec or GloVe) as sentence embeddings. 

To get GloVe embeddings, run (in examples/):
```bash
./get_glove.bash
```

To reproduce the results for avg(GloVe) vectors, run :  
```bash
python examples/bow.py
```

Logistic regression in pytorch is quite fast, though for inner cross-validation it takes a bit of time to converge since it has to learn a model k * k_inner * #grid.

As required by SentEval, this script implements two functions : **batcher** (required), and **prepare** (optional) that turn text sentences into sentence embeddings.

The user transforms text sentences into embeddings using these functions, and SentEval takes care of the rest : evaluating the quality of these embeddings on any task above.

## How SentEval works

To evaluate your own model, you just need to implement two functions : 

1. batcher
2. prepare

that will transform bach of sentences coming from transfer tasks to batch of embeddings.
On these functions are implemented, the SentEval framework will use these embeddings as *pre-trained* features and learn (linear/nonlinear) models on top of them to perform the tasks above. The user can choose between Logistic Regression or Multi-Layer Perceptron (MLP) (see [here](https://arxiv.org/pdf/1705.02364.pdf) for more details on the models).

### 1.) batcher(batch, params)
```
batcher(batch, params)
```
* *batch* : numpy array of text sentences (of size params.batch_size)
* *params* : senteval parameters (note that "prepare" outputs are stored in params).
* *output* : numpy array of sentence embeddings (of size params.batch_size)

In bow.py : [*params.word_vect[word]* for word in batch[0]] can be used to extract the word vectors of the first sentence in batch.
The mean of the word vectors is then computed.

### 2.) prepare(params, samples)

batcher only sees one batch at a time. To create a vocabulary of the whole dataset (for instance for SNLI), we need to see the whole dataset.

```
prepare(params, samples)
```
* *batch* : numpy array of text sentences
* *params* : senteval parameters (note that "prepare" outputs are stored in params).
* *output* : None. Any "output" computed in this function is stored in "params" and can be further used by *batcher*.

allows the user to build a vocabulary (chars, subwords or words), usually named *word2id* and *id2word*.

In bow.py, it is used to extract the "params.word_vect* dictionary of word vectors. It c

It can also be used to initialize the lookup table of a neural network using GloVe vectors.



### 3.) evaluation on transfer tasks

Once you've implemented your own batcher and functions using your sentence encoder, you are ready to evaluate your encoder on transfer tasks.

To perform the actual evaluation, first import senteval and define a SentEval object :
```python
import senteval
se = senteval.SentEval(batcher, prepare, params_senteval)
```
define the set of transfer tasks on which you want SentEval to perform evaluation and run the evaluation : 
```python
transfer_tasks = ['MR', 'MPQA', 'SST', 'TREC', 'SICKRelatedness', 'SICKEntailment', 'MRPC', 'ImageAnnotation']
results = se.eval(transfer_tasks)
```

This simple interface namely allows to evaluate the quality of a sentence encoder while training (at every epoch).

## SentEval parameters
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
