from setuptools import find_packages, setup
from typing import List 

def get_requirements(file_path: str) -> List[str]:
    """This function will return a list of requirements."""
    requirements = []
    try:
        with open(file_path, encoding="utf-8") as file_obj:
            requirements = file_obj.readlines()
            requirements = [req.strip() for req in requirements if req.strip()]
            if "-e ." in requirements:
                requirements.remove("-e .")
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. No dependencies will be installed.")
    return requirements

setup(
    name="MLPROJECT",
    version="0.0.1",
    author="Tanishq Anand",
    author_email="tanishqanand26@gmail.com", 
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)