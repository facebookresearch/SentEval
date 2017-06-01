# SentEval

SentEval is a library for evaluating the quality of sentence embeddings as features for a broad and diverse set of "transfer" tasks. It is aimed to ease the study and the development of better general-purpose fixed-size sentence representations (see [InferSent](https://arxiv.org/pdf/1705.02364.pdf)).

## Dependencies

This code is written in python. The dependencies are :

* Python 2.7
* [Pytorch](http://pytorch.org/) >= 0.12
* [NumPy](http://www.numpy.org/) and [SciPy](http://www.scipy.org/)
* [scikit-learn](http://scikit-learn.org/stable/index.html)>="0.18.0"


## Tasks

See [here](https://arxiv.org/pdf/1705.02364.pdf) for a detailed description of the available tasks.
* Binary classification : MR (movie review), CR (product review), SUBJ (subjectivity status), MPQA (opinion-polarity), SST (sentiment analysis)
* Multi-class classification : TREC (question-type classification), SST (fine-grained sentiment analysis)
* Entailment (NLI) and semantic relatedness : [SNLI](https://nlp.stanford.edu/projects/snli/) (entailment), [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/) (entailment), [SICK](http://clic.cimec.unitn.it/composes/sick.html) (entailment/relatedness)
* Semantic Textual Similarity : [STSBenchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark#Results), [STS14](http://alt.qcri.org/semeval2014/task10/)
* Paraphrase detection : [MRPC](https://aclweb.org/aclwiki/index.php?title=Paraphrase_Identification_(State_of_the_art))
* Caption-Image retrieval : [COCO](http://mscoco.org/) dataset (with ResNet-101 2048d image embeddings)


## Download datasets
```bash
python data/get_transfer_data.py --data_path your_output_path
```
will automatically download the dataset and preprocess them to "your_output_path".

WARNING : Downloading the [MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398) dataset requires the "[cabextract](https://www.cabextract.org.uk/#install)" command line (sudo yum install cabextract/sudo apt-get isntall cabextract) to extract the provided Microsoft-specific MSI files. In data/get_transfer_data.py, you can decide to keep or skip the MRPC dataset.

## How-to example : examples/bow.py - average(word2vec)

### examples/bow.py
In examples/bow.py, we provide a minimal example for evaluating the quality of the *average of word vectors* (word2vec or GloVe) as sentence embeddings. 

To reproduce the results for avg(GloVe) vectors, run :  
```bash
python examples/bow.py
```

As required by SentEval, this script implements two functions : **batcher** (required), and **prepare** (optional) that turn text sentences into sentence embeddings.

The user transforms text sentences into embeddings using these functions, and SentEval takes care of the rest : evaluating the quality of these embeddings on any task above.

## How-to

To evaluate your own model, you just need to implement two functions : 

1. batcher
2. prepare

that will transform bach of sentences coming from transfer tasks to batch of embeddings

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
* *task_path* (str) : path to data, generated by data/get_transfer_data.py
* *seed* (int) : random seed for reproducability (default : 1111)
* *usepytorch* (bool) : use pytorch or scikit learn for logistic regression (default : True)
* *batch_size* (int) : size of minibatch of text sentences provided to "batcher" (sentences are sorted by length) (note that this is not the batch_size used by pytorch logistic regression, which is fixed)
* *verbose* (int) : 2->debug, 1->info, 0->warning (default : 2)
* ... and any parameter you want to have access to in "batcher" or "prepare" functions.


## TODO
* Remove "network" parameter in batcher (the user can put it in "params") **DONE**
* In bow.py, remove the lookup table (get_lut_glove). Create a word_vects dictionnary of word vectors. **DONE**
* Reduce the number of parameters of senteval.SentEval **DONE**
* Add SST/multi, add MultiNLI in get_transfer_data
* Add SST/bin on the github. Add COCO dataset on the github (Amazon S3 with proper clearance from legal to redistribute?)
* Provide IDF weights in .pickle and propose average-idf-weighted-wordvectors ?
* data download in bash file **DONE**
* preprocess MRPC directly in download file and remove NLTK dependency

