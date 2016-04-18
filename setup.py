#!/usr/bin/env python

from setuptools import setup


# extract version from __init__.py
with open('s2sphere/__init__.py', 'r') as f:
    version_line = [l for l in f if l.startswith('__version__')][0]
    VERSION = version_line.split('=')[1].strip()[1:-1]


setup(
    name='s2sphere',
    version=VERSION,
    description='Python implementation of the amazing S2 Geometry Library',
    long_description=open('README.md').read(),
    author='Jonathan Gillham',
    author_email='',
    url='http://github.com/qedus/sphere',
    packages=['s2sphere'],
    install_requires=['future'],
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 33',
        'Programming Language :: Python :: 2',
    ],
)
