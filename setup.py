from setuptools import setup, find_packages

setup(name='frankenstein',
    version='0.0.1',
    install_requires=[
        'torch',
        'tensordict',
        'jaxtyping',
        'tabulate',
        'gymnasium>=1.0.0',
        'typing-extensions',
        'packaging', # Required by tensordict but not listed in its dependencies for some reason
    ],
    extras_require={
        'dev': ['pytest','pytest-cov','pytest-timeout','pdoc','numpy','flake8','autopep8'],
        'benchmark': ['gymnasium[mujoco]', 'ale-py']
    },
    packages=find_packages()
)
