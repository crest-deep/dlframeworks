#!/usr/bin/env python
from setuptools import setup

setup(
    name='dlframeworks',
    version='0.1',
    description='Tools for Deep Learning frameworks',
    author='Yohei TSUJI',
    author_email='tsuji.y.ae@m.titech.ac.jp',
    package=[
        'dlframeworks',
        'dlframeworks.chainer',
        'dlframeworks.chainer.optimizers',
        'dlframeworks.chainer.utils',
        'dlframeworks.tensorflow',
    ],
    zip_safe=False,
    install_requires=[
        'matplotlib',
        'numpy',
    ]
)
