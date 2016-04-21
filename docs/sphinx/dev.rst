.. _dev:


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

.. code-block:: sh

    # point to Brew's OpenSSL installation
    export OPENSSL_ROOT_DIR=$(brew --prefix openssl)

    # tell the Python cmake which libraries to use
    cmake -DPYTHON_LIBRARY=$(python-config --prefix)/lib/libpython2.7.dylib -DPYTHON_INCLUDE_DIR=$(python-config --prefix)/include/python2.7 .
