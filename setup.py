from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str) -> List[str]:
    """
    This Function will return a list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements if HYPHEN_E_DOT not in req]
    return requirements

setup(
    name = 'MLproject',
    version = '0.0.1',
    author = 'anfeher',
    author_email = 'ahdez0905@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)