from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'


def get_requirements(file_path: str) -> List[str]:
    '''
    This func will return requirements list
    '''
    requirements = []
    with open(file_path, 'r') as f:
        requirements = f.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements      

setup(
    name='mlproject',
    version='0.0.1',
    author='Jinay Panchal',
    author_email='panchaljinay11@gmail.com',
    packages=find_packages(),
    install_requirements=get_requirements('requirements.txt')


)
