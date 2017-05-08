from setuptools import setup, find_packages

# This reads the __version__ variable from projectq/_version.py
exec(open('src/fermilib/_version.py').read())
# Readme file as long_description:
long_description = open('README').read()
# Read in requirements.txt
requirements = open('requirements.txt').readlines()
requirements = [r.strip() for r in requirements]


setup(
    name='fermilib',
    version=__version__,
    author='Ryan Babbush, Jarrod McClean, Damian Steiger, Ian Kivlichan, '
    'Thomas Haener, Vojtech Havlicek, Matthew Neeley, Wei Sun',
    author_email='ryanbabbush@gmail.com, jarrod.mcc@gmail.com, '
                 'fermilib@projectq.ch',
    url='http://www.projectq.ch',
    description=('FermiLib - '
                 'An open source package for analyzing, compiling and '
                 'emulating quantum algorithms for simulation of fermions.'),
    long_description=long_description,
    install_requires=requirements,
    license='Apache 2',
    packages=find_packages(where='src'),
    package_dir={'': 'src'}
)
