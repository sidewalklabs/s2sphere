#!/usr/bin/env python
import os

from setuptools import setup, find_packages


def read(f):
    return open(os.path.join(os.path.dirname(__file__), f)).read().strip()


setup(name='s2-sphere',
      version='0.1.0',
      description='Python implementation of the amazing S2 Geometry Library',
      long_description=read('README.md'),
      author='Jonathan Gillham',
      author_email='',
      url='http://github.com/qedus/sphere',
      packages=find_packages(),
      install_requires=['future'],
      license='MIT',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Programming Language :: Python :: 33',
          'Programming Language :: Python :: 2',
      ],
)
