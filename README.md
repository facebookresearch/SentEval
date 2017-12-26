# SentEval: evaluation toolkit for sentence embeddings

SentEval is a library for evaluating the quality of sentence embeddings. We assess their generalization power by using them as features on a broad and diverse set of "transfer" tasks. **SentEval currently includes 17 tasks**. Our goal is to ease the study and the development of general-purpose fixed-size sentence representations.


**SentEval recent fixes (12/26):**
1. renamed main directory: new way to import SentEval (see below)
2. fixed **reproducibility issue for STS tasks** (which was due to issue in data preprocessing get_transfer_data)
3. now single download file in data/
4. added option to **set parameters of the classifier** (nhid, optim, lr, batch size, ...)
5. added example of classifier setting to **speed up training (x5) in prototyping phase**

**New Transfer Tasks Coming Soon ...**

## Dependencies

This code is written in python. The dependencies are:

* Python 2.7 (with [NumPy](http://www.numpy.org/)/[SciPy](http://www.scipy.org/))
* [Pytorch](http://pytorch.org/) >= 0.2
* [scikit-learn](http://scikit-learn.org/stable/index.html)>=0.18.0

## Transfer tasks

SentEval allows you to evaluate your sentence embeddings as features for the following tasks:

| Task     	| Type                         	| #train 	| #test 	| needs_train 	| set_classifier |
|----------	|------------------------------	|-----------:|----------:|:-----------:|:----------:|
| [MR](https://nlp.stanford.edu/~sidaw/home/projects:nbsvm)       	| movie review                 	| 11k     	| 11k    	| 1 | 1 |
| [CR](https://nlp.stanford.edu/~sidaw/home/projects:nbsvm)       	| product review               	| 4k      	| 4k     	| 1 | 1 |
| [SUBJ](https://nlp.stanford.edu/~sidaw/home/projects:nbsvm)     	| subjectivity status          	| 10k     	| 10k    	| 1 | 1 |
| [MPQA](https://nlp.stanford.edu/~sidaw/home/projects:nbsvm)     	| opinion-polarity  | 11k     	| 11k    	| 1 | 1 |
| [SST](https://nlp.stanford.edu/sentiment/index.html)      	| (binary) sentiment analysis  	| 67k     	| 1.8k   	| 1 | 1 |
| [TREC](http://cogcomp.cs.illinois.edu/Data/QA/QC/)     	| question-type classification 	| 6k      	| 0.5k    	| 1 | 1 |
| [SICK-E](http://clic.cimec.unitn.it/composes/sick.html)   	| recognizing textual entailment 	| 4.5k    	| 4.9k   	| 1 | 1 |
| [SNLI](https://nlp.stanford.edu/projects/snli/)     	| natural language inference   	| 550k    	| 9.8k   	| 1 | 1 |
| [MRPC](https://aclweb.org/aclwiki/Paraphrase_Identification_(State_of_the_art)) | paraphrase detection  | 4.1k | 1.7k | 1 | 1 |
| [STS 2012](https://www.cs.york.ac.uk/semeval-2012/task6/) 	| semantic textual similarity  	| N/A     	| 3.1k   	| 0  | 0 |
| [STS 2013](http://ixa2.si.ehu.es/sts/) 	| semantic textual similarity  	| N/A     	| 1.5k   	| 0  | 0 |
| [STS 2014](http://alt.qcri.org/semeval2014/task10/) 	| semantic textual similarity  	| N/A     	| 3.7k   	| 0  | 0 |
| [STS 2015](http://alt.qcri.org/semeval2015/task2/) 	| semantic textual similarity  	| N/A     	| 8.5k   	| 0  | 0 |
| [STS 2016](http://alt.qcri.org/semeval2016/task1/) 	| semantic textual similarity  	| N/A     	| 9.2k   	| 0  | 0 |
| [STS B](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark#Results)    	| semantic textual similarity  	| 5.7k    	| 1.4k   	| 1 | 0 |
| [SICK-R](http://clic.cimec.unitn.it/composes/sick.html)   	| semantic textual similarity | 4.5k    	| 4.9k   	| 1 | 0 |
| [COCO](http://mscoco.org/)     	| image-caption retrieval      	| 567k    	| 5*1k   	| 1 | 0 |

where **needs_train** means a model with parameters is learned on top of the sentence embeddings, and **set_classifier** means you can define the parameters of the classifier in the case of a classification task (see below).

Note: COCO comes with ResNet-101 2048d image embeddings. [More details on the tasks.](https://arxiv.org/pdf/1705.02364.pdf)

## Download datasets
To get all the transfer tasks datasets, run (in data/):
```bash
./get_transfer_data_ptb.bash
```
This will automatically download and preprocess the datasets, and store them in data/senteval_data (warning: for MacOS users, you may have to use p7zip instead of unzip). Note: we provide PTB or MOSES tokenization.

WARNING: Extracting the [MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398) MSI file requires the "[cabextract](https://www.cabextract.org.uk/#install)" command line (i.e *apt-get/yum install cabextract*).

## How to use SentEval: examples

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

## How to use SentEval

To evaluate your sentence embeddings, SentEval requires that you implement two functions:

1. **prepare** (sees the whole dataset of each task and can thus construct the word vocabulary, the dictionary of word vectors etc)
2. **batcher** (transforms a batch of text sentences into sentence embeddings)


### 1.) prepare(params, samples) (optional)

*batcher* only sees one batch at a time while the *samples* argument of *prepare* contains all the sentences of a task.

```
prepare(params, samples)
```
* *params*: senteval parameters.
* *samples*: list of all sentences from the tranfer task.
* *output*: No output. Arguments stored in "params" can further be used by *batcher*.

*Example*: in bow.py, prepare is is used to build the vocabulary of words and construct the "params.word_vect* dictionary of word vectors.


### 2.) batcher(params, batch)
```
batcher(params, batch)
```
* *params*: senteval parameters.
* *batch*: numpy array of text sentences (of size params.batch_size)
* *output*: numpy array of sentence embeddings (of size params.batch_size)

*Example*: in bow.py, batcher is used to compute the mean of the word vectors for each sentence in the batch using params.word_vec. Use your own encoder in that function to encode sentences.

### 3.) evaluation on transfer tasks

After having implemented the batch and prepare function for your own sentence encoder,

1) to perform the actual evaluation, first import senteval and set its parameters:
```python
import senteval
params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
```

2) (optional) set the parameters of the classifier (when applicable):
```python
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}
```
You can choose **nhid=0** (Logistic Regression) or **nhid>0** (MLP) and define the parameters for training.

3) Create an instance of the class SE:
```python
se = senteval.engine.SE(params, batcher, prepare)
```

4) define the set of transfer tasks and run the evaluation:
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

## SentEval parameters (fast version)
SentEval has several parameters (only task_path is required):
Global parameters of SentEval:
```bash
# senteval parameters
task_path                   # path to SentEval datasets
seed                        # seed
usepytorch                  # use cuda-pytorch (else scikit-learn) where possible
kfold                       # k-fold validation for MR/CR/SUB/MPQA.
```

Parameters of the classifier (by default nonlineary is Tanh for MLP):
```bash
nhid:                       # number of hidden units (0: Logistic Regression)
optim:                      # optimizer ("sgd,lr=0.1", "adam", "rmsprop" ..)
tenacity:                   # how many times dev acc does not increase before stopping
epoch_size:                 # each epoch corresponds to epoch_size pass on the train set
max_epoch:                  # max number of epoches
dropout:                    # dropout for MLP
```

Note that to get a proxy of the results while **dramatically reducing computation time**,
we suggest the **prototyping config**:
```python
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}
```
which will results in a 5 times speedup for classification tasks.

To produce results that are **comparable to the literature**, use the **default config**:
```python
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}
```
which takes longer but will produce better and comparable results.

## References

Please considering citing [[1]](https://arxiv.org/abs/1705.02364) [[2]](https://arxiv.org/abs/1707.06320) if using this code for evaluating sentence embedding methods.

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

Contact: [aconneau@fb.com](mailto:aconneau@fb.com), [dkiela@fb.com](mailto:dkiela@fb.com)
