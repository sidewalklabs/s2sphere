#!/usr/bin/env python

from setuptools import setup


# extract version from __init__.py
with open('s2sphere/__init__.py', 'r') as f:
    version_line = [l for l in f if l.startswith('__version__')][0]
    VERSION = version_line.split('=')[1].strip()[1:-1]


setup(
    name='s2sphere',
    version=VERSION,
    description='Python implementation of the S2 Geometry Library',
    long_description=open('README.rst').read(),
    author='Jonathan Gillham and contributors',
    author_email='sven@sidewalklabs.com',
    url='http://s2sphere.readthedocs.io',
    packages=['s2sphere'],
    install_requires=['future'],
    extras_require={
        'tests': [
            'flake8>=2.5.4',
            'hacking>=0.11.0',
            'nose>=1.3.4',
            'numpy>=1.11.0'
        ],
        'docs': [
            'Sphinx>=1.4.1',
            'sphinx-rtd-theme>=0.1.9'
        ]
    },
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],
)
