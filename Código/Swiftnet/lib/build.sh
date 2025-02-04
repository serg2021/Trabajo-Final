#!/usr/bin/env bash
#rm cylib.so

cython -a cylib.pyx -o cylib.cc

g++ -shared -pthread -fPIC -std=c++11 -fwrapv -O3 -Wall -fno-strict-aliasing \
-I/home/sergios/miniconda2/envs/swiftnet3/lib/python3.7/site-packages/numpy/core/include -I/home/sergios/miniconda2/envs/swiftnet3/include/python3.7m -o cylib.so cylib.cc
#g++ -shared -pthread -fPIC -fwrapv -O3 -Wall -fno-strict-aliasing \
#-I/usr/lib/python3.8/site-packages/numpy/core/include -I/usr/include/python3.8 -o cylib.so cylib.cc
