from setuptools import setup, find_packages
from os import path

HERE = path.split(path.abspath(__file__))[0]


requirements = []

with open(path.join(HERE, 'requirements.txt'), 'r') as f:
    for line in f.readlines():
        if not line.startswith('#'):
            requirements.append(line.strip())


setup(name='self_supervised',
      install_requires=requirements,
      packages=find_packages(),
      version='0.1'
)
