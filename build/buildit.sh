#!/bin/bash

COLLECTION="europarl"
COLLECTION_FILE="data/europarl-v7.de-en.en"

# COLLECTION="toy"
# COLLECTION_FILE="data/toyfile2.txt"

echo "Removing collection $COLLECTION" && rm -rf ../collections/$COLLECTION && echo "Creating collection $COLLECTION from $COLLECTION_FILE" && ./create-collection.x -i $COLLECTION_FILE -c ../collections/$COLLECTION  && echo "Building index $COLLECTION" && ./build-index.x -c ../collections/$COLLECTION/ -m
