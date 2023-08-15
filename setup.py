from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT='-e .'
def get_requirements(filepath:str)->List[str]:
    '''Get requirements from requirements.txt file'''
    with open(filepath, 'r') as f:
        requirements = f.readlines()
        requirements = [r.replace("\n","") for r in requirements if r != HYPHEN_E_DOT]
    return requirements
setup(
    name='score-predictor',
    version='0.0.1',
    author='Alex',
    author_email='vuxminhan@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    )

