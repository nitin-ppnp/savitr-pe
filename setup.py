#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='savitr_pe',
      version='0.0.1',
      description='Savitr pose estimation',
      author='Nitin Saini',
      author_email='nitin.ppnp@gmail.com',
      url='',
      install_requires=[],
      packages=find_packages('src'),
      package_dir={'':'src'},
      )