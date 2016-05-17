s2sphere
========

.. image:: https://badge.fury.io/py/s2sphere.svg
    :target: https://pypi.python.org/pypi/s2sphere/
.. image:: https://travis-ci.org/sidewalklabs/s2sphere.svg?branch=master
    :target: https://travis-ci.org/sidewalklabs/s2sphere
.. image:: https://coveralls.io/repos/github/sidewalklabs/s2sphere/badge.svg?branch=master
    :target: https://coveralls.io/github/sidewalklabs/s2sphere?branch=master

Python implementation of a part of the C++ `S2 geometry library <https://code.google.com/p/s2-geometry-library/>`_.

Install with:

.. code-block:: sh

    pip install s2sphere


Links: `Documentation <http://s2sphere.readthedocs.io>`_,
`GitHub <https://github.com/sidewalklabs/s2sphere>`_,
`Issue Tracker <https://github.com/sidewalklabs/s2sphere/issues>`_

To set up the library for development:

.. code-block:: sh

    pip install .[tests]
    nosetests -vv --exclude=compare_implementations_test
