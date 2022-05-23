from setuptools import setup
from os import path

HERE = path.split(path.abspath(__file__))[0]


requirements = []

with open(path.join(HERE, 'requirements.txt'), 'r') as f:
    for line in f.readlines():
        if not line.startswith('#'):
            requirements.append(line.strip())


setup(install_requires=requirements)