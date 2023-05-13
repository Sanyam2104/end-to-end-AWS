from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(FilePath:str)->List[str]:
    '''
        This function reads the requirements.txt file and returns a 
        list of the packages written in it.
    '''
    requirements = []

    with open(FilePath) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n','') for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements


setup(
    name='credit_card_approval', 
    version='1.1',
    author='Sanyam',
    author_email='sanyamjain1154@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')   
)