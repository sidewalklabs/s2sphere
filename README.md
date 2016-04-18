sphere
======

[![Build Status](https://travis-ci.org/sidewalklabs/sphere.svg?branch=modularize)](https://travis-ci.org/sidewalklabs/sphere)

Python implementation of the amazing C++ [S2 Geometry Library](https://code.google.com/p/s2-geometry-library/). The S2 Geometry Library is explained in more detail [here](https://docs.google.com/presentation/d/1Hl4KapfAENAOf4gv-pSngKwvS_jwNVHRPZTTDzXXn6Q/view).

It basically maps a sphere to a 1D index. This allows you to do scalable proximity searches on distributed indexes such as with MongoDB and App Engine Datastore. It also has a load of other excellent features for dealing with spheres. I am yet to find a better system for mapping spheres to 1D indexes.

The tests are quite extensive and reflect those in the original S2 Geometry Library.


Tests
=====

Tests also build the original C++ library. Building tests on OSX:
* `export OPENSSL_ROOT_DIR=$(brew --prefix openssl)`
* `cmake -DPYTHON_LIBRARY=$(python-config --prefix)/lib/libpython2.7.dylib -DPYTHON_INCLUDE_DIR=$(python-config --prefix)/include/python2.7 .`


TODO
====

* A few more S2 features could be added especially those associated with segments of a sphere.
* I regret some of the naming conventions and wish they were more in line with how the C++ S2 library was originally packaged. For example I think it might be nicer for there to be separate packages called r1, s1 and s2. This would make also make it conform better with the mathematics behind the S2 Geometry Library.
