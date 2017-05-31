"""
Automatically downloads data for both training and transfer tasks
Alexis Conneau (2017)
"""


import os
import argparse

parser = argparse.ArgumentParser(description='get all data for senteval')
parser.add_argument('--data_path', required=False, help='path', default='senteval_data')
parser.add_argument('--mrpc', required=False, default=True)
args = parser.parse_args()

data_path = args.data_path
os.system('mkdir ' + data_path)
preprocess_exec = './tokenizer.sed'

### Define paths to datasets
transfer_paths = {}
transfer_paths['TREC'] = []
transfer_paths['TREC'].append('http://cogcomp.cs.illinois.edu/Data/QA/QC/train_5500.label')
transfer_paths['TREC'].append('http://cogcomp.cs.illinois.edu/Data/QA/QC/TREC_10.label')

transfer_paths['SICK'] = []
transfer_paths['SICK'].append('http://alt.qcri.org/semeval2014/task1/data/uploads/sick_train.zip')
transfer_paths['SICK'].append('http://alt.qcri.org/semeval2014/task1/data/uploads/sick_trial.zip')
transfer_paths['SICK'].append('http://alt.qcri.org/semeval2014/task1/data/uploads/sick_test_annotated.zip')

transfer_paths['CLASSIF1'] = 'http://www.stanford.edu/~sidaw/projects/datasmall_NB_ACL12.zip'

transfer_paths['STS14'] = 'http://alt.qcri.org/semeval2014/task10/data/uploads/sts-en-gs-2014.zip'
transfer_paths['STSBenchmark'] = 'http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz'
                    
# MRPC links to msi files that are not so easy to parse with UNIX
transfer_paths['MRPC'] = 'https://download.microsoft.com/download/D/4/6/D46FF87A-F6B9-4252-AA8B-3604ED519838/MSRParaphraseCorpus.msi'

transfer_paths['SNLI'] = 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'

transfer_paths['COCO'] = [] # TODO : Add these files to the github (with copyright mentions)
transfer_paths['COCO'].append('https://github.com/fair/sent2vec/coco/train.pkl')
transfer_paths['COCO'].append('https://github.com/fair/sent2vec/coco/valid.pkl')
transfer_paths['COCO'].append('https://github.com/fair/sent2vec/coco/test.pkl')


### download STS2014 and STSBenchmark (http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark)
# STS14
os.system('mkdir ' + os.path.join(data_path, 'STS'))
os.system('wget -O - ' + transfer_paths['STS14'] + ' > ' + os.path.join(data_path, 'data_sts.zip'))
os.system('unzip ' + os.path.join(data_path, 'data_sts.zip') + ' -d ' + os.path.join(data_path))
os.system('mv ' + os.path.join(data_path, 'sts-en-test-gs-2014') + ' ' + os.path.join(data_path, 'STS', 'STS14'))
os.system('rm ' + os.path.join(data_path, 'data_sts.zip'))

for sts_task in ['deft-forum', 'deft-news', 'headlines', 'OnWN', 'images', 'tweet-news']:
    fname = "STS.input." + sts_task + ".txt"
    os.system('cut -f1 ' + os.path.join(data_path, 'STS', 'STS14', fname) + ' | ' + preprocess_exec + ' > ' + os.path.join(data_path, 'STS', 'STS14', 'tmp1'))
    os.system('cut -f2 ' + os.path.join(data_path, 'STS', 'STS14', fname) + ' | ' + preprocess_exec + ' > ' + os.path.join(data_path, 'STS', 'STS14', 'tmp2'))
    os.system('paste ' + os.path.join(data_path, 'STS', 'STS14', 'tmp1') + ' ' + os.path.join(data_path, 'STS', 'STS14', 'tmp2') + ' > ' + os.path.join(data_path, 'STS', 'STS14', fname))
    os.system('rm ' + os.path.join(data_path, 'STS', 'STS14', 'tmp1') + ' ' + os.path.join(data_path, 'STS', 'STS14', 'tmp2'))
        
# STSBenchmark
os.system('wget -O - ' + transfer_paths['STSBenchmark'] + ' > ' + os.path.join(data_path, 'Stsbenchmark.tar.gz'))
os.system('tar -zxvf ' + os.path.join(data_path, 'Stsbenchmark.tar.gz') + ' -C ' + data_path)
os.system('rm ' + os.path.join(data_path, 'Stsbenchmark.tar.gz'))
os.system('mv ' + os.path.join(data_path, 'stsbenchmark') + ' ' + os.path.join(data_path, 'STS', 'STSBenchmark'))
if True:
    for split in ['train', 'dev', 'test']:
        fname = 'sts-' + split + '.csv'
        fdir = os.path.join(data_path, 'STS', 'STSBenchmark')
        os.system('cut -f1,2,3,4,5 ' + os.path.join(fdir, fname) + ' > ' + os.path.join(fdir, 'tmp1'))
        os.system('cut -f6 ' + os.path.join(fdir, fname) + ' | ' +  preprocess_exec  + ' > ' + os.path.join(fdir, 'tmp2'))
        os.system('cut -f7 ' + os.path.join(fdir, fname) + ' | ' +  preprocess_exec  + ' > ' + os.path.join(fdir, 'tmp3'))
        os.system('paste ' + os.path.join(fdir, 'tmp1') + ' ' + os.path.join(fdir, 'tmp2') + ' ' + os.path.join(fdir, 'tmp3') + ' > ' + os.path.join(fdir, fname))
        os.system('rm ' + os.path.join(fdir, 'tmp1') + ' ' + os.path.join(fdir, 'tmp2') + ' ' + os.path.join(fdir, 'tmp3'))


### download TREC
for transfer_task in ['TREC']:
    os.system('mkdir ' + os.path.join(data_path, transfer_task))
    for path in transfer_paths[transfer_task]:
        print 'yo'
        print path
        print  os.path.join(data_path, transfer_task, path.split('/')[-1])
        os.system('wget -O - ' + path + ' > ' + os.path.join(data_path, transfer_task, path.split('/')[-1]) )
        
### download SICK
for transfer_task in ['SICK']:
    os.system('mkdir ' + os.path.join(data_path, transfer_task))
    for path in transfer_paths[transfer_task]:
        os.system('wget -O - ' + path + ' > ' + os.path.join(data_path, transfer_task, path.split('/')[-1]) )
        os.system('unzip ' + os.path.join(data_path, transfer_task, path.split('/')[-1]) + ' -d ' + os.path.join(data_path, transfer_task))
        os.system('rm ' + os.path.join(data_path, transfer_task, path.split('/')[-1]))        

### download MR CR SUBJ MPQA
# Download and unzip file
os.system('wget -O - ' + transfer_paths['CLASSIF1'] + ' > ' + os.path.join(data_path, 'data_classif.zip'))
os.system('unzip ' + os.path.join(data_path, 'data_classif.zip') + ' -d ' + os.path.join(data_path, 'data_bin_classif'))

# MR
os.system('mkdir ' + os.path.join(data_path, 'MR'))
os.system('cat ' + os.path.join(data_path, 'data_bin_classif/data/rt10662/rt-polarity.pos') + ' | ' + preprocess_exec + ' > ' + os.path.join(data_path, 'MR', 'rt-polarity.pos'))
os.system('cat ' + os.path.join(data_path, 'data_bin_classif/data/rt10662/rt-polarity.neg') + ' | ' + preprocess_exec + ' > ' + os.path.join(data_path, 'MR', 'rt-polarity.neg'))

# CR
os.system('mkdir ' + os.path.join(data_path, 'CR'))
os.system('cat ' + os.path.join(data_path, 'data_bin_classif/data/customerr/custrev.pos') + ' | ' + preprocess_exec + ' > ' + os.path.join(data_path, 'CR', 'custrev.pos'))
os.system('cat ' + os.path.join(data_path, 'data_bin_classif/data/customerr/custrev.neg') + ' | ' + preprocess_exec + ' > ' + os.path.join(data_path, 'CR', 'custrev.neg'))


# SUBJ
os.system('mkdir ' + os.path.join(data_path, 'SUBJ'))
os.system('cat ' + os.path.join(data_path, 'data_bin_classif/data/subj/subj.subjective') + ' | ' + preprocess_exec + ' > ' + os.path.join(data_path, 'SUBJ', 'subj.subjective'))
os.system('cat ' + os.path.join(data_path, 'data_bin_classif/data/subj/subj.objective') + ' | ' + preprocess_exec + ' > ' + os.path.join(data_path, 'SUBJ', 'subj.objective'))

# MPQA
os.system('mkdir ' + os.path.join(data_path, 'MPQA'))
os.system('cat ' + os.path.join(data_path, 'data_bin_classif/data/mpqa/mpqa.pos') + ' | ' + preprocess_exec + ' > ' + os.path.join(data_path, 'MPQA', 'mpqa.pos'))
os.system('cat ' + os.path.join(data_path, 'data_bin_classif/data/mpqa/mpqa.neg') + ' | ' + preprocess_exec + ' > ' + os.path.join(data_path, 'MPQA', 'mpqa.neg'))

# CLEAN-UP
os.system('rm ' + os.path.join(data_path, 'data_classif.zip'))
os.system('rm -r ' + os.path.join(data_path, 'data_bin_classif'))
        
### download SNLI
os.system('mkdir ' + os.path.join(data_path, 'SNLI'))
os.system('wget -O - ' + transfer_paths['SNLI'] + ' > ' + os.path.join(data_path, 'SNLI', 'snli_1.0.zip'))
os.system('unzip ' + os.path.join(data_path, 'SNLI', 'snli_1.0.zip') + ' -d ' + os.path.join(data_path, 'SNLI'))
os.system('rm ' + os.path.join(data_path, 'SNLI', 'snli_1.0.zip'))
os.system('rm -r ' + os.path.join(data_path, 'SNLI', '__MACOSX'))


os.system("awk '{ if ( $1 != \"-\" ) { print $0; } }' " + os.path.join(data_path, 'SNLI', 'snli_1.0', 'snli_1.0_train.txt') + " | cut -f 1,6,7 | sed '1d' > " + os.path.join(data_path, 'SNLI', 'train.snli.txt'))
os.system("awk '{ if ( $1 != \"-\" ) { print $0; } }' " + os.path.join(data_path, 'SNLI', 'snli_1.0', 'snli_1.0_dev.txt') + " | cut -f 1,6,7 | sed '1d' > " + os.path.join(data_path, 'SNLI', 'dev.snli.txt'))
os.system("awk '{ if ( $1 != \"-\" ) { print $0; } }' " + os.path.join(data_path, 'SNLI', 'snli_1.0', 'snli_1.0_test.txt') + " | cut -f 1,6,7 | sed '1d' > " + os.path.join(data_path, 'SNLI', 'test.snli.txt'))

for split in ["train", "dev", "test"]:
    fpath = os.path.join(data_path, 'SNLI', split + '.snli.txt')
    os.system('cut -f1 ' + fpath + ' > ' + os.path.join(data_path, 'SNLI', 'labels.' + split))
    os.system('cut -f2 ' + fpath + ' | ' + preprocess_exec + ' > ' + os.path.join(data_path, 'SNLI', 's1.' + split))
    os.system('cut -f3 ' + fpath + ' | ' + preprocess_exec + ' > ' + os.path.join(data_path, 'SNLI', 's2.' + split))
    os.system('rm ' + fpath)
os.system('rm -r ' + os.path.join(data_path, 'SNLI', 'snli_1.0'))



"""
TODO : Put COCO and SST on the github and provide a download link
TODO : how to make the user install "cabextract" properly?
"""
if False:
    ### download COCO
    os.system('mkdir ' + os.path.join(data_path, 'COCO'))
    cocopath = '/mnt/vol/gfsai-east/ai-group/users/aconneau/projects/sentence-encoding/transfer-tasks/COCO/coco5k/'
    os.system('cp ' + os.path.join(cocopath, 'train.pkl') + ' ' +  os.path.join(data_path, 'COCO', 'train.pkl'))
    os.system('cp ' + os.path.join(cocopath, 'valid.pkl') + ' ' +  os.path.join(data_path, 'COCO', 'valid.pkl'))
    os.system('cp ' + os.path.join(cocopath, 'test.pkl') + ' ' +  os.path.join(data_path, 'COCO', 'test.pkl'))

    ### download SST
    os.system('mkdir ' + os.path.join(data_path, 'SST'))
    sstpath = '/mnt/vol/gfsai-east/ai-group/users/aconneau/projects/sentence-encoding/transfer-tasks/SST/'
    os.system('cp -r ' + sstpath + ' ' +  data_path)

    ### download MRPC
    # This extraction needs "cabextract" to extract the MSI file
    # os.system('sudo yum install cabextract')

    os.system('sudo yum install cabextract')
    os.system('mkdir ' + os.path.join(data_path, 'MRPC'))
    os.system('wget -O - ' + transfer_paths['MRPC'] + ' > ' + os.path.join(data_path, 'MRPC', 'MSRParaphraseCorpus.msi'))

    os.system('cabextract ' + os.path.join(data_path, 'MRPC', 'MSRParaphraseCorpus.msi') + ' -d ' + os.path.join(data_path, 'MRPC'))

    os.system('cat ' + os.path.join(data_path, 'MRPC', '_2DEC3DBE877E4DB192D17C0256E90F1D') + ' > ' + os.path.join(data_path, 'MRPC', 'msr_paraphrase_train.txt'))
    os.system('cat ' + os.path.join(data_path, 'MRPC', '_2D65ED66D69C42A28B021C3E24C1D8C0') + ' > ' + os.path.join(data_path, 'MRPC', 'msr_paraphrase_data.txt'))
    os.system('cat ' + os.path.join(data_path, 'MRPC', '_D7B391F9EAFF4B1B8BCE8F21B20B1B61') + ' > ' + os.path.join(data_path, 'MRPC', 'msr_paraphrase_test.txt'))

    # CLEAN-UP
    os.system('rm ' + os.path.join(data_path, 'MRPC', '_*'))
    os.system('rm ' + os.path.join(data_path, 'MRPC', 'MSRParaphraseCorpus.msi'))


    # CLEAN-UP
    os.system('rm ' + os.path.join(data_path, 'MRPC', '_*'))
    os.system('rm ' + os.path.join(data_path, 'MRPC', 'MSRParaphraseCorpus.msi'))
