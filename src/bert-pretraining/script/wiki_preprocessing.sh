#!/bin/bash

# step 1 unzip the wiki file

name=wiki18
src=/dfs/scratch1/mleszczy/data/wiki_2018/20190128
# preprocess using wikiextractor.py
mkdir -p ../../../data/wiki/$name

pushd ../../../data/wiki/$name

echo "start copying"
cp -r $src/enwiki-latest-pages-articles.xml.bz2 ./

echo "start unzip"
bzip2 -d enwiki-latest-pages-articles.xml.bz2 &

pushed ../
git clone https://github.com/attardi/wikiextractor.git
pushed wikiextractor
git checkout 3162bb6
popd
popd

echo "start extracting"
python ../wikiextractor/WikiExtractor.py --processes 100 -o ./wiki_json --json enwiki-latest-pages-articles.xml &
# python ../wikiextractor/WikiExtractor.py --processes 30 -o ./wiki_xml enwiki-latest-pages-articles.xml
echo "extraction done "
popd