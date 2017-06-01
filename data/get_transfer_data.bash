# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

data_path=senteval_data
preprocess_exec=./tokenizer.sed

mkdir $data_path

TREC=()
TREC+=('http://cogcomp.cs.illinois.edu/Data/QA/QC/train_5500.label')
TREC+=('http://cogcomp.cs.illinois.edu/Data/QA/QC/TREC_10.label')

SICK=()
SICK+=('http://alt.qcri.org/semeval2014/task1/data/uploads/sick_train.zip')
SICK+=('http://alt.qcri.org/semeval2014/task1/data/uploads/sick_trial.zip')
SICK+=('http://alt.qcri.org/semeval2014/task1/data/uploads/sick_test_annotated.zip')

BINCLASSIF='http://www.stanford.edu/~sidaw/projects/datasmall_NB_ACL12.zip'

STS14='http://alt.qcri.org/semeval2014/task10/data/uploads/sts-en-gs-2014.zip'

STSBenchmark='http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz'

# MRPC links to msi files that are not so easy to parse with UNIX
MRPC='https://download.microsoft.com/download/D/4/6/D46FF87A-F6B9-4252-AA8B-3604ED519838/MSRParaphraseCorpus.msi'

SNLI='https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
MULTINLI='https://www.nyu.edu/projects/bowman/multinli/multinli_0.9.zip'

COCO=()
COCO+=('https://github.com/fair/sent2vec/coco/train.pkl')
COCO+=('https://github.com/fair/sent2vec/coco/valid.pkl')
COCO+=('https://github.com/fair/sent2vec/coco/test.pkl')



### download STS2014 and STSBenchmark (http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark)
# STS14
echo $data_path/STS
mkdir $data_path/STS
curl -o $data_path/data_sts.zip $STS14
unzip $data_path/data_sts.zip -d $data_path
mv $data_path/sts-en-test-gs-2014 $data_path/STS/STS14
rm $data_path/data_sts.zip

for sts_task in deft-forum deft-news headlines OnWN images tweet-news
do
    fname=STS.input.$sts_task.txt
    cut -f1 $data_path/STS/STS14/$fname | ./tokenizer.sed > $data_path/STS/STS14/tmp1
    cut -f2 $data_path/STS/STS14/$fname | ./tokenizer.sed > $data_path/STS/STS14/tmp2
    paste $data_path/STS/STS14/tmp1 $data_path/STS/STS14/tmp2 > $data_path/STS/STS14/$fname
    rm $data_path/STS/STS14/tmp1 $data_path/STS/STS14/tmp2
done

# STSBenchmark
curl -o $data_path/Stsbenchmark.tar.gz $STSBenchmark
tar -zxvf $data_path/Stsbenchmark.tar.gz -C $data_path
rm $data_path/Stsbenchmark.tar.gz
mv $data_path/stsbenchmark $data_path/STS/STSBenchmark

for split in train dev test
do
    fname=sts-$split.csv
    fdir=$data_path/STS/STSBenchmark
    cut -f1,2,3,4,5 $fdir/$fname > $fdir/tmp1
    cut -f6 $fdir/$fname | $preprocess_exec > $fdir/tmp2
    cut -f7 $fdir/$fname | $preprocess_exec > $fdir/tmp3
    paste $fdir/tmp1 $fdir/tmp2 $fdir/tmp3 > $fdir/$fname
    rm $fdir/tmp1 $fdir/tmp2 $fdir/tmp3
done


### download TREC
mkdir $data_path/TREC
for path in ${TREC[@]}
do
    curl -o $data_path/TREC/$(basename $path) $path
done

### download SICK
mkdir $data_path/SICK
for path in ${SICK[@]}
do
    curl -o $data_path/SICK/$(basename $path) $path
    unzip $data_path/SICK/$(basename $path) -d $data_path/SICK/
    rm $data_path/SICK/readme.txt
    rm $data_path/SICK/$(basename $path)
done


### download MR CR SUBJ MPQA
# Download and unzip file
curl -o $data_path/data_classif.zip $BINCLASSIF
unzip $data_path/data_classif.zip -d $data_path/data_bin_classif
rm $data_path/data_classif.zip

# MR
mkdir $data_path/MR
cat $data_path/data_bin_classif/data/rt10662/rt-polarity.pos | $preprocess_exec > $data_path/MR/rt-polarity.pos
cat $data_path/data_bin_classif/data/rt10662/rt-polarity.neg | $preprocess_exec > $data_path/MR/rt-polarity.neg

# CR
mkdir $data_path/CR
cat $data_path/data_bin_classif/data/customerr/custrev.pos | $preprocess_exec > $data_path/CR/custrev.pos
cat $data_path/data_bin_classif/data/customerr/custrev.neg | $preprocess_exec > $data_path/CR/custrev.neg

# SUBJ
mkdir $data_path/SUBJ
cat $data_path/data_bin_classif/data/subj/subj.subjective | $preprocess_exec > $data_path/SUBJ/subj.subjective
cat $data_path/data_bin_classif/data/subj/subj.objective | $preprocess_exec > $data_path/SUBJ/subj.objective

# MPQA
mkdir $data_path/MPQA
cat $data_path/data_bin_classif/data/mpqa/mpqa.pos | $preprocess_exec > $data_path/MPQA/mpqa.pos
cat $data_path/data_bin_classif/data/mpqa/mpqa.neg | $preprocess_exec > $data_path/MPQA/mpqa.neg

# CLEAN-UP
rm -r $data_path/data_bin_classif

### download SNLI
mkdir $data_path/SNLI
curl -o $data_path/SNLI/snli_1.0.zip $SNLI
unzip $data_path/SNLI/snli_1.0.zip -d $data_path/SNLI
rm $data_path/SNLI/snli_1.0.zip
rm -r $data_path/SNLI/__MACOSX

for split in train dev test
do
    fpath=$data_path/SNLI/$split.snli.txt
    awk '{ if ( $1 != "-" ) { print $0; } }' $data_path/SNLI/snli_1.0/snli_1.0_$split.txt | cut -f 1,6,7 | sed '1d' > $fpath
    cut -f1 $fpath > $data_path/SNLI/labels.$split
    cut -f2 $fpath | $preprocess_exec > $data_path/SNLI/s1.$split
    cut -f3 $fpath | $preprocess_exec > $data_path/SNLI/s2.$split
    rm $fpath
done
rm -r $data_path/SNLI/snli_1.0


### download MRPC
# This extraction needs "cabextract" to extract the MSI file

sudo yum install cabextract
sudo brew install cabextract

mkdir $data_path/MRPC
curl -o $data_path/MRPC/MSRParaphraseCorpus.msi $MRPC
cabextract $data_path/MRPC/MSRParaphraseCorpus.msi -d $data_path/MRPC
# ****HACK**** renaming files
cat $data_path/MRPC/_2DEC3DBE877E4DB192D17C0256E90F1D | tr -d $'\r' > $data_path/MRPC/msr_paraphrase_train.txt
cat $data_path/MRPC/_D7B391F9EAFF4B1B8BCE8F21B20B1B61 | tr -d $'\r' > $data_path/MRPC/msr_paraphrase_test.txt
rm $data_path/MRPC/_*
rm $data_path/MRPC/MSRParaphraseCorpus.msi

for split in train test
do
    fname=$data_path/MRPC/msr_paraphrase_$split.txt
    cut -f1,2,3 $fname | sed '1d' > $data_path/MRPC/tmp1
    cut -f4 $fname | sed '1d' | $preprocess_exec > $data_path/MRPC/tmp2
    cut -f5 $fname | sed '1d' | $preprocess_exec > $data_path/MRPC/tmp3
    head -n 1 $fname > $data_path/MRPC/tmp4
    paste $data_path/MRPC/tmp1 $data_path/MRPC/tmp2 $data_path/MRPC/tmp3 >> $data_path/MRPC/tmp4
    mv $data_path/MRPC/tmp4 $fname
    rm $data_path/MRPC/tmp1 $data_path/MRPC/tmp2 $data_path/MRPC/tmp3
done

# TODO : COCO and SST



