from setuptools import setup, find_packages

setup(
    name='Contextuality',
    version='0.0.1',
    url='https://github.com/kinianlo/contextuality',
    author='Kin Ian Lo',
    author_email='keonlo123@gmail.com',
    description='A package used to handle empirical models.',
    packages=find_packages(),    
    install_requires=['numpy', 'picos'],
)
