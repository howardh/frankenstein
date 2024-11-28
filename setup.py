from setuptools import setup, find_packages

setup(name='frankenstein',
    version='0.0.1',
    install_requires=['torch', 'tensordict', 'torchtyping', 'tabulate', 'gymnasium>=1.0.0'],
    extras_require={
        'dev': ['pytest','pytest-cov','pytest-timeout','pdoc','numpy','flake8','autopep8'],
        'benchmark': ['gymnasium[mujoco]', 'ale-py']
    },
    packages=find_packages()
)
