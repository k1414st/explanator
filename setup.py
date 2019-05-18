# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='explanator',
    version='0.0.1',
    description='explanation tools of models',
    long_description=readme,
    author='k1414st',
    author_email='k1414st@gmail.com',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    # install_requires=['tensorflow', 'keras'],
    install_requires=[],
)

