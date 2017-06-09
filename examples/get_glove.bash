glovepath='http://nlp.stanford.edu/data/glove.840B.300d.zip'

echo $glovepath
mkdir glove
curl -LO $glovepath
unzip glove.840B.300d.zip -d glove/
rm glove.840B.300d.zip

