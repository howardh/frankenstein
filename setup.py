from setuptools import setup, find_packages

setup(name='frankenstein',
    version='0.0.1',
    install_requires=['torch', 'torchtyping'],
    extras_require={
        'dev': ['pytest','pytest-cov','pytest-timeout','pdoc','numpy','flake8','autopep8'],
    },
    packages=find_packages()
)
