import os

from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop

# This reads the __version__ variable from projectq/_version.py
exec(open('src/fermilib/_version.py').read())
# Readme file as long_description:
long_description = open('README').read()
# Read in requirements.txt
requirements = open('requirements.txt').readlines()
requirements = [r.strip() for r in requirements]


def post_install(directory):
    path = os.path.join(directory, 'data')
    print('Using {} as data directory.'.format(str(path)))


class ExtendedInstall(install):
    def run(self):
        install.run(self)
        self.execute(post_install, (self.install_headers,),
                     msg="Configuring FermiLib")


class ExtendedDevelop(develop):
    def run(self):
        develop.run(self)
        self.execute(post_install, (self.egg_path,),
                     msg="Configuring FermiLib")


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
    cmdclass={'install': ExtendedInstall,
              'develop': ExtendedDevelop},
    license='Apache 2',
    packages=find_packages(where='src'),
    package_dir={'': 'src'}
)
