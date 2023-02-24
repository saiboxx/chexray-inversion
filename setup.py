"""A setuptools-based script for installing the chexray-inversion package."""

from setuptools import find_packages, setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='',
    long_description=long_description,
    long_description_content_type='text/markdown',
)
