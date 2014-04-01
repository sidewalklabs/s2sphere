sphere
======

Python implementation of the amazing C++ [S2 Geometry Library](https://code.google.com/p/s2-geometry-library/). The S2 Geometry Library is explained in more detail [here](https://docs.google.com/presentation/d/1Hl4KapfAENAOf4gv-pSngKwvS_jwNVHRPZTTDzXXn6Q/view).

It basically maps a sphere to a 1D index. This allows you to do scalable proximity searches on distributed indexes such as with MongoDB and App Engine Datastore. It also has a load of other excellent features for dealing with spheres. I am yet to find a better system for mapping spheres to 1D indexes.

The tests are quite extensive and reflect those in the original S2 Geometry Library.

TODO
====

* Needs to be packaged properly for use with PIP.
* A few more S2 features could be added especially those associated with segments of a sphere.
* I regret some of the naming conventions and wish they were more in line with how the C++ S2 library was originally packaged. For example I think it might be nicer for there to be separate packages called r1, s1 and s2. This would make it more inline with the mathematics behind the S2 Geometry Library.
