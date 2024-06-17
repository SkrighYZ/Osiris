#!/bin/bash

# modified from https://gist.github.com/moskomule/2e6a9a463f50447beca4e64ab4699ac4

wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -P $DATA_DIR
cd $DATA_DIR
unzip tiny-imagenet-200.zip

current="${DATA_DIR}/tiny-imagenet-200"

# training data
echo "restructuring train"
cd $current/train
for DIR in $(ls); do
   cd $DIR
   rm *.txt
   mv images/* .
   rm -r images
   cd ..
done

# validation data
echo "restructuring val"
cd $current/val
annotate_file="val_annotations.txt"
length=$(cat $annotate_file | wc -l)
for i in $(seq 1 $length); do
    # fetch i th line
    line=$(sed -n ${i}p $annotate_file)
    # get file name and directory name
    file=$(echo $line | cut -f1 -d" " )
    directory=$(echo $line | cut -f2 -d" ")
    mkdir -p $directory
    mv images/$file $directory
done
rm -r images
echo "done"