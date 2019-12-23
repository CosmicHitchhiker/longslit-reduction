#!/usr/bin/env python3

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    req = f.read().splitlines()

setup(
    name='longslit',
    version='0.0.1',
    author='Vsevolod Lander',
    author_email='sevalander@gmail.com',
    description='Pipeline for longslit observation data reduction',
    url='https://github.com/CosmicHitchhiker/longslit-reduction',
    license='MIT',
   # package_dir={'': ''},
    packages=find_packages(where='longslit'),
    scripts=['longslit/pipeline.py'],
    entry_points={
        'console_scripts': ['longslit-pipeline = pipeline:main'],
    },
    test_suite='test',
    install_requires=req,
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Topic :: Education',
        'Programming Language :: Python :: 3',
        # See full list on https://pypi.org/classifiers/
    ],
    keywords='reduction observation spectra',
)