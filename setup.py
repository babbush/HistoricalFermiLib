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
    author='TODO',
    author_email='TODO',
    url='http://www.projectq.ch',
    description=('FermiLib - '
                 'WRITE MORE'),
    long_description=long_description,
    install_requires=requirements,
    license='Apache 2',
    packages=find_packages(where='src'),
    package_dir={'': 'src'}
)
