from setuptools import setup
import os
import sys

meta = {}
with open("bladex/meta.py") as fp:
    exec(fp.read(), meta)

# Package meta-data.
NAME = meta['__title__']
DESCRIPTION = 'BladeX.'
URL = 'https://github.com/mathLab/BladeX'
MAIL = meta['__mail__']
AUTHOR = meta['__author__']
VERSION = meta['__version__']
KEYWORDS='blade-generation propeller iges procal'

REQUIRED = [
    'numpy', 'scipy', 'matplotlib', 'Sphinx', 'sphinx_rtd_theme',
]

def readme():
    """
    This function just return the content of README.md
    """
    with open('README.md') as f:
        return f.read()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=readme(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    keywords=KEYWORDS,
    url=URL,
    author=AUTHOR,
    author_email=MAIL,
    license='MIT',
    packages=[NAME],
    install_requires=REQUIRED,
    test_suite='nose.collector',
    tests_require=['nose'],
    include_package_data=True,
    zip_safe=False)