from setuptools import find_packages, setup

HYPEN_E_DOT = '-e .'
def get_requirements(file_path: str) -> list:
    with open(file_path) as file_obj:
        lines = file_obj.readlines()
        lines = [lines.replace('\n', '') for lines in lines]

        if HYPEN_E_DOT in lines:
            lines.remove(HYPEN_E_DOT)

    return lines


setup(
    name = 'ml_project',
    version = '0.0.1',
    author = 'Kartik',
    author_email = 'kartik99jais@gmail.com',
    packages=find_packages(),
    install_requires=[
        get_requirements('requirements.txt')
    ],
)