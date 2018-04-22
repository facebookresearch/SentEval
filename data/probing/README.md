# Probing tasks

## General remarks

All data sets contain 100k training instances, 10k validation instances and 10k test instances, and in all cases they are balanced across the target classes (in some cases, there are a few more instances, as a result of balancing constraints in the sampling process). Each instance is on a separate line, and contains (at least) the following tab-separated fields:

- the first field specifies the partition (tr/va/te);

- the second field specifies the ground-truth class of the instance (e.g., PRES/PAST in the past_present.txt file);

- the last field contains the sentence (in space-delimited tokenized format).

Fields between the third and the next-to-last contain information that is useful to build and analyze the relevant data-set: in some cases, it might be sensible to preserve this information in the version we distribute (as discussed below on a case-by-case basis)

In all data sets, the instances are ordered by partition, but, within each partition, they are randomized.

I describe below the filtering steps that were applied to the parsed corpus in order to generate a specific data-set. We should however also add to this readme a description of the general, preliminary corpus-preprocessing steps we performed upstream.

The source sentences were taken from the [BookCorpus](http://yknzhu.wixsite.com/mbweb), and more precisely from the pre-processed subset found in the training partition of [Lambada](http://clic.cimec.unitn.it/lambada/).

Note that we relied on the [Stanford Parser](https://nlp.stanford.edu/software/lex-parser.shtml) for part of speech, dependency and constituency parsing information. We used the 2017-06-09 version of the parser, relying on the pre-trained PCFG model. We post-processed the parser output to make it compatible with the [Moses](http://www.statmt.org/moses/) tokenization conventions, since the latter are assumed by the SentEval tools.

## top_constituents.txt

This is a 20-class classification task, where the classes are given by the 19 most common top-constituent sequences in the corpus, plus a 20th category for all other structures. The classes are:

- ADVP_NP_VP_.
- CC_ADVP_NP_VP_.
- CC_NP_VP_.
- IN_NP_VP_.
- NP_ADVP_VP_.
- NP_NP_VP_.
- NP_PP_.
- NP_VP_.
- OTHER
- PP_NP_VP_.
- RB_NP_VP_.
- SBAR_NP_VP_.
- SBAR_VP_.
- S_CC_S_.
- S_NP_VP_.
- S_VP_.
- VBD_NP_VP_.
- VP_.
- WHADVP_SQ_.
- WHNP_SQ_.

Top-constituent sequences that contained sentence-internal punctuation marks and quotes or did not end with the . label were excluded (also from the OTHER class).

As expected, there is some variation in sentence length across classes. Looking specifically at the test partition, the NP_PP_. top constituent has the shortest median length (6 words), whereas S_NP_VP. has the longest (18). The mode is between 11 and 12 words. It would be good to include a baseline model that uses sentence length as the only feature to predict the classes, to get an estimate of how far this heuristic can go.

The file contains 4 fields, with field #3 reporting sentence length. This can be removed from the public version of the data set, since it can trivially be extracted from the tokenized-sentence field.

## past_present.txt

This is a binary classification task, based on whether the main verb of the sentence is marked as being in the present or past tense. The present tense corresponds to PoS tags VBP and VBZ, whereas the past tense corresponds to VBD.

Only sentences where the main verb has a corpus frequency of between 100 and 5,000 occurrences are considered. More importantly, a verb form can only occur in one of the partitions. For example, the past form "provided" only occurs in the training set.

There is a small tendency for past forms to be more frequent than present forms, but this seems negligible, especially on a log-frequency scale (e.g., difference between median log frequencies in the test set is 0.32). Still, we might want to report a baseline that uses target verb form frequency as the only feature (such frequency should be collected on a full corpus, not from the benchmark itself). Note that here, and for the tasks below, such baseline would assume access to the structure of the sentence, in order to correctly identify the verb of the main clause, so it is far from a trivial baseline.

There is also a small tendency for sentences with past forms to be longer than those with present forms, but again this seems negligible (13 vs 11 medians). Again, a sentence-length baseline might be considered.

The file contains 5 fields. Field #3 contains the target verb form, field #4 contains sentence length. Target verb form might be preserved in the public version, as it might be useful for further analysis (or for baselines that rely on features of the target verb).

## subj_number.txt

Another binary classification task, this time focusing on the number of the subject of the main clause (this should generally also be the main verb number, but number in English is more systematically marked on nouns than verbs). The classes are  NN (singular) and NNS (plural or mass: "personnel", "clientele", etc). As the class labels suggest, only common nouns are considered.

Like above, only target noun forms with corpus frequency between 100 and 5,000 are considered, and noun forms are split across the partitions.

There is a small tendency for singular forms to be more frequent than plural ones (0.61 difference between median log frequencies in the test set), whereas the two classes have the same sentence length distributions. It might thus be desirable to include a frequency baseline.

Format is the same as for the previous file.

## obj_number.txt

Same as above, but this time focusing on the direct object of the main clause. See the description of the previous task for labels, filtering and format. We observe a rather small difference in frequency between singular and plural forms (0.21 difference between median log frequencies in the test set). There is no difference in terms of sentence length. Again, a frequency baseline might provide a good control.

## odd_man_out.txt

This binary task asks whether a sentence occurs as-is in the source corpus (O label, for Original), or whether a (single) randomly picked noun or verb was replaced with another form with the same part of speech (C label, for Changed). To make the task interesting, the original word and the replacement have comparable frequencies for the bigrams they form with the immediately preceding and following tokens. More precisely, the frequencies of these bigrams have to be in the same frequency range, with range boundaries logarithmically set at 1, 10, 100, and 1000 (e.g., a bigram with frequency 15 is in the same range of, and thus compatible with, a bigram with frequency 79).

We only selected for replacement nouns and verbs, since random replacements from other classes (most notably: adjectives) often have vague meanings that were semantically acceptable in the targeted contexts.

Both target and replacement were filtered to have corpus frequency between 40 and 400 occurrences. This range is considerably lower than for the other data sets, because very frequent words tend to have vague meanings that are compatible with many contexts.

More importantly, for the sentences with replacement, the replacement words only occur in one partition. Moreover, no sentence occurs in both the original and changed versions.

There is a very small tendency for replacement words to be more frequent than the originals (0.13 difference between log median frequencies in the test set).  Note however that in the case of the originals, it is actually arbitrary to consider the word that was NOT replaced as the one we measure the frequency of, so a frequency baseline makes no sense in this case. There is no difference in sentence length.

For this data set, it would be useful to report some human performance data. I plan to annotate 200 test sentences manually, but I'd do that only after we are sure we have a stable version of the set.

The file contains 12 fields. Field #3 is an arbitrary, unique sentence id. Field #4 indexes the position of the replacement in the sentence. Field #5 is the PoS of the replacement. Fields #6 and #7 contain the target and its replacement, respectively, and fields #8 and #9 their respective frequencies. Fields #10 and #11 contain the ranges of the bigrams formed with the preceding and following words, respectively (denoted by their upper bounds). Note that, potentially confusingly, all fields are present for original sentences as well, since this information was useful for benchmark creation (i.e., I picked a candidate replacement for each sentence, irrespective of whether I would actually use it). The extra fields that might be usefully made publicly available, for changed sentences only, are original, replacement, their position, PoS, and perhaps bigram ranges.

Finally, note that, due to the random sampling procedure used to pick the data, there are very small asymmetries in the class sizes. These should be negligible, as in no partition the larger class is represented by more than 50.2% of the instances.

## coordination_inversion.txt

Binary task asking to distinguish between original sentence (class O) and sentences where the order of two coordinated clausal conjoints has been inverted (class I). An example of the latter is: "There was something to consider but he might be a prince". Only sentences that contain only one coordinating conjunction (CC) are considered, and the conjunction must coordinate two top clauses (sentences must match one of the following top-constituent patterns: S_CC_S_., S_,\_CC_S_. or S_:\_CC_S_.).

In constructing the data set, we balanced the sentences by the length of the two conjoined clauses, that is, both the original and inverted sets contain an equal number of cases in which the first clause is longer, the second one is longer, and they are of equal length.

Less importantly, no sentence is presented in both original and inverted order. Also, not surprisingly, there is no difference in sentence length distribution between the two classes.

Here, too, it would be good to report human performance, based on a subset of the test data, once they become stable.

The file contains 6 fields. Field #2 is an arbitrary, unique sentence id, and fields #3 and #4 contain the lengths of the first and second conjoints in the underlying original sentence. None of these fields seems generally useful, and they should be removed from the public version.

## sentence_length.txt

This is a classification task where the goal is to predict the sentence length which has been binned in 6 possible categories with lengths ranging in the following intervals: --0: (5-8), 1: (9-12), 2: (13-16), 3: (17-20), 4: (21-25), 5: (26-29). These are the same bins from Adi et al. except for the two larger ones --(30-33), (34-70)-- which we excluded because they were filtered out in our corpus pre-processing step. 

Each of the bins have equal number of examples.

## tree_depth.txt

This is a classification tasks where the goal is to predict the depth of the sentence's syntactic tree at its maximium depth (with values ranging from 5 to 12). 

For this task, in the raw corpus distribution, the maximum tree depth correlates a lot with sentence length (>0.8 ). For this reason, we have run the following procedure to choose a partially decorrelated sample. First, we picked a range of sentence lengths and sentence depths delimited by the 10 and 90 percentiles of each of the two distributions.  That is, sentences whose tree depths are between 5 and 12 and their lengths between 6 and 23. There we set as target a (cropped) multivariate gaussian with mean centered on the empirical mean of the data (8.35 sentence depth and 13.2 sentence length) and diagonal variances 10 (sentence depth) and 30 (sentence length). Finally we we extract 143328 samples following the desired distribution. The obtained sample has a pearson coefficient between depth and length of r=0.1 (p highly significant), from which we randomly sample 100k for training, 10k for testing and 10k for validation. 

Finally, we tested the top syntactic trees for high tree depths (>= 9) and low depths (<=7) in the extracted dataset, noticing that this may still be confound, but yet NP-VP is still by far the most common structure.


## word_content.txt

This is a classification task with 1000 words as targets. The task is predicting which of the target words appear on the given sentence.

We constructed the data by picking the first 1000 lower-cased words occurring on rank >2k having length of at least 4 characters (to remove words like “d” or “y”). Then shuffled the sentences and picked 100 training examples per word, making sure that there is only one word from the candidate present in the sentence, and 10 + 10 validation and testing examples.


## bigram_shift.txt

In this classification task the goal is to predict whether two consecutive tokens within the sentence have been inverted (1 for inversion, 0 for original).

We constructed this data by choosing at random two consecutive tokens in the sentence, making sure that we don't pick the first two tokens in the sentence to avoid capitalization being a queue. We also excluded sentences containing double quotation and, for similar reasons, we didn't perform invertion with punctuation marks.
