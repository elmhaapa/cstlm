Welcome to CSTLM
================

This is a fork of the [CSTLM by Shareghi et al](https://github.com/eehsan/cstlm), which I used as a part of my M.Sc. thesis. If you want to use this code, please refer to the work of the original authors.


## Compile instructions

1. Check out the reprository: `https://github.com/elmhaapa/cstlm.git`
3. `git checkout dev-branch`
4. `git submodule update --init`
5. `cd build`
6. `cmake ..`
7. `make -j`

## Run unit tests to ensure correctness

```
cd build
rm -rf ../collections/unittest/
./create-collection.x -i ../UnitTestData/data/training.data -c ../collections/unittest
./create-collection.x -i ../UnitTestData/data/training.data -c ../collections/unittest -1
./unit-test.x
```

## Usage instructions (Word based language model)

Create collection:

```
./create-collection.x -i toyfile.txt -c ../collections/toy
```

Build index (including quantities for modified KN)

```
./build-index.x -c ../collections/toy/ -m
```

Query index (i.e., Modified KN (drop -m for KN), 5-gram)

```
./query-index-knm.x -c ../collections/toy/ -p test.txt -m -n 5 
```
## Usage instructions (Character based language model)

Create collection:

```
./create-collection.x -i toyfile.txt -c ../collections/toy -1
```

Build index (including quantities for modified KN)

```
./build-index.x -c ../collections/toy/ -m
```

## Experiments

Different branches contain different experiments:

- **dev-branch** has a functioning version of the previous work by Shareghi et al.
- **hac-vector** uses hacs instead of dacs.
- **buffering-vector-struct** contains the buffering approach.
- **bwt-iter** has the iterating bwt approach. 


