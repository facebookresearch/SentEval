-- Copyright 2004-present Facebook. All Rights Reserved.

SentEval is a python tool for evaluating the quality of  distributed sentence representations (embeddings) 


See "examples/bow.py" for an example.
SentEval requires that you implement two functions : 
    1) batcher
    2) prepare


It then handles the validation of your embeddings on a set of transfer tasks this way:
    se = senteval.SentEval(...)
    results = se.eval(['MR', 'SICK' ...])

On these tasks, a classifier will be learned on top of your embeddings (LogReg-default or MLP).


*** List of tasks ***
MR, CR, SUBJ, MPQA : binary classification
SST : Stanford Sentiment Treebank binary classification dataset.
MRPC : Paraphrase detection
TREC : Question-type classification
SICKRelatedness : Predict a similarity score for two sentences.
SICKEntailment : Recognizing textual entailment (RTE).
SNLI : Entailment big dataset.
STS14 : Semantic textual similarity tasks (unsupervised).
ImageAnnotation : COCO image annotation and image search task.

# SentEval
