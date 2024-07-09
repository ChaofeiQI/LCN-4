#!/usr/bin/env bash

echo "Dataset:" $1  "Backbone:" $2  "GPU index:" $3 "Tag:" $4 
#cd ../
python exp1_train_classifier.py --gpu $3 --config ./configs/$1/$2-$1.yaml --tag=$4