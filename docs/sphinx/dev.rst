.. _dev:


Developing
==========

To develop, clone the repository and include git submodules recursively:

.. code-block:: sh

    git clone --recursive https://github.com/sidewalklabs/s2sphere

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

.. code-block:: sh

    # point to Brew's OpenSSL installation
    export OPENSSL_ROOT_DIR=$(brew --prefix openssl)

    # tell the Python cmake which libraries to use
    cmake -DPYTHON_LIBRARY=$(python-config --prefix)/lib/libpython2.7.dylib -DPYTHON_INCLUDE_DIR=$(python-config --prefix)/include/python2.7 .

Then install this module with the dependencies needed for running tests and
generating docs:

.. code-block:: sh

    # install
    pip install -e .[tests,docs]

    # run tests without C lib
    flake8
    nosetests -vv --exclude=compare_implementations_test


Documentation
-------------

.. code-block:: sh

    cd docs/sphinx
    make html


Tests
-----

.. code-block:: sh

    nosetests -vv
