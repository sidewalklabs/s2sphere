s2sphere
========

.. image:: https://travis-ci.org/sidewalklabs/sphere.svg?branch=modularize
    :target: https://travis-ci.org/sidewalklabs/sphere

Python implementation of the amazing C++ `S2 Geometry Library <https://code.google.com/p/s2-geometry-library/>`_. The S2 Geometry Library is explained in more detail `here <https://docs.google.com/presentation/d/1Hl4KapfAENAOf4gv-pSngKwvS_jwNVHRPZTTDzXXn6Q/view>`_.

It basically maps a sphere to a 1D index. This allows you to do scalable proximity searches on distributed indexes such as with MongoDB and App Engine Datastore. It also has a load of other excellent features for dealing with spheres. I am yet to find a better system for mapping spheres to 1D indexes.

The tests are quite extensive and reflect those in the original S2 Geometry Library for the part that is implemented in this module.


Getting Started
===============

.. code-block:: sh

    pip install s2sphere  # not yet
    pip install https://github.com/sidewalklabs/sphere/archive/modularize.zip


For example, to get the S2 cells covering a LatLon-rectangle:

.. code-block:: sh

    import s2sphere

    r = s2sphere.RegionCoverer()
    p1 = s2sphere.LatLon.from_degrees(33, -122)
    p2 = s2sphere.LatLon.from_degrees(33.1, -122.1)
    cell_ids = r.get_covering(s2sphere.LatLonRect.from_point_pair(p1, p2))
    print(cell_ids)

which prints this list:

.. code-block:: sh

    [9291041754864156672, 9291043953887412224, 9291044503643226112, 9291045878032760832, 9291047252422295552, 9291047802178109440, 9291051650468806656, 9291052200224620544]


Developing
==========

To develop, clone the repository and include git submodules recursively:

.. code-block:: sh

    git clone --recursive https://github.com/sidewalklabs/sphere

Tests require ``pip install numpy`` and a build of the original C++ library:

.. code-block:: sh

    # build and install the C++ library
    cd tests/s2-geometry/geometry
    cmake .
    make -j4
    make install

    # build the C++ library's Python bindings
    cd python
    cmake .  # see comment below for OSX
    make -j4
    make install

    # verify Python bindings
    python -v -c 'import s2'


OSX requires extra setup:

- point to Brew's OpenSSL installation: ``export OPENSSL_ROOT_DIR=$(brew --prefix openssl)``
- tell the Python cmake which Python libraries to use: ``cmake -DPYTHON_LIBRARY=$(python-config --prefix)/lib/libpython2.7.dylib -DPYTHON_INCLUDE_DIR=$(python-config --prefix)/include/python2.7 .``
