.. s2sphere documentation master file, created by
   sphinx-quickstart on Wed Apr 20 14:08:01 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


s2sphere
========

.. image:: https://travis-ci.org/sidewalklabs/s2sphere.svg?branch=master
    :target: https://travis-ci.org/sidewalklabs/s2sphere

Python implementation of a part of the C++ `S2 geometry library <https://code.google.com/p/s2-geometry-library/>`_. Install with:

.. code-block:: sh

    pip install s2sphere

The S2 geometry library is explained in more detail in
`this presentation by Octavian Procopiuc <https://docs.google.com/presentation/d/1Hl4KapfAENAOf4gv-pSngKwvS_jwNVHRPZTTDzXXn6Q/view>`_.
It maps a sphere to a 1D index. This enables scalable proximity searches using
distributed indices such as with MongoDB and App Engine Datastore. The test
cases of this package are extensive and reflect those in the original S2
geometry library.

Links: `Documentation <http://s2sphere.readthedocs.io>`_,
`GitHub <https://github.com/sidewalklabs/s2sphere>`_,
`Issue Tracker <https://github.com/sidewalklabs/s2sphere/issues>`_


Contents
--------

.. toctree::
    :maxdepth: 2

    quickstart
    api
    dev

* :ref:`genindex`

