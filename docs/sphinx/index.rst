.. s2sphere documentation master file, created by
   sphinx-quickstart on Wed Apr 20 14:08:01 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


s2sphere
========

.. image:: https://travis-ci.org/sidewalklabs/sphere.svg?branch=modularize
    :target: https://travis-ci.org/sidewalklabs/sphere

Python implementation of the amazing C++ `S2 Geometry Library <https://code.google.com/p/s2-geometry-library/>`_. The S2 Geometry Library is explained in more detail `here <https://docs.google.com/presentation/d/1Hl4KapfAENAOf4gv-pSngKwvS_jwNVHRPZTTDzXXn6Q/view>`_.

Install with:

.. code-block:: sh

    pip install s2sphere  # not yet
    pip install https://github.com/sidewalklabs/sphere/archive/modularize.zip

It basically maps a sphere to a 1D index. This allows you to do scalable proximity searches on distributed indexes such as with MongoDB and App Engine Datastore. It also has a load of other excellent features for dealing with spheres. I am yet to find a better system for mapping spheres to 1D indexes.

The tests are quite extensive and reflect those in the original S2 Geometry Library.



Contents
--------

.. toctree::
   :maxdepth: 2

   quickstart
   api

* :ref:`genindex`

