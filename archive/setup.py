#!usr/bin/env python
# -*- coding: utf-8 -*-

"""This is the setupy.py"""

from setuptools import setup, find_packages

setup(
    name='Power_system_split',
    version='0.0.1',
    author=['Franz Kaiser', 'Philipp C. Böttcher'],
    author_email=['f.kaiser@fz-juelich.de'
                  'p.boettcher@fz-juelich.de'],
    description='Analyse relevant power system splits after failure cascades.',
    packages=find_packages(),
    long_description=open('README.md').read(),
    install_requires=open('requirements.txt').read().splitlines(),
)
