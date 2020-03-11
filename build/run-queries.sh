#!/bin/bash

COLLECTION="europarl"
FILE="data/europarl-tail-10000-fi-en.en"
# COLLECTION="toy"
# FILE="testi.txt"

for i in {1..10}; do
  echo "Running query against collection $COLLECTION with file: $FILE"
  echo "Running query for m length $i"
  ./query-index-knm.x -c ../collections/$COLLECTION -p $FILE -m -n $i
done
