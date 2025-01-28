from setuptools import setup, find_packages

meta = {}
with open("bladex/meta.py") as fp:
    exec(fp.read(), meta)

# Package meta-data.
NAME = meta['__title__']
DESCRIPTION = 'Python Blade Morphing'
URL = 'https://github.com/mathLab/BladeX'
MAIL = meta['__mail__']
AUTHOR = meta['__author__']
VERSION = meta['__version__']
KEYWORDS = 'blade-generation propeller iges procal'

REQUIRED = [
    'numpy', 'scipy', 'matplotlib', 'Sphinx', 'sphinx_rtd_theme', 'smithers'
]

EXTRAS = {
    'docs': ['Sphinx==1.4', 'sphinx_rtd_theme'],
    'test': ['pytest', 'pytest-cov'],
}

LDESCRIPTION = (
    "BladeX is a Python package for geometrical parametrization and bottom-up "
    "construction of propeller blades. It allows to generate and deform a "
    "blade based on the radial distribution of its parameters such as pitch, "
    "rake, skew, and the sectional foils' parameters such as chord and "
    "camber. The package is ideally suited for parametric simulations on "
    "large number of blade deformations. It provides an automated procedure "
    "for the CAD generation, hence reducing the time and effort required for "
    "modelling. The main scope of BladeX is to deal with propeller blades, "
    "however it can be flexible to be applied on further applications with "
    "analogous geometrical structures such as aircraft wings, turbomachinery, "
    "or wind turbine blades."
)

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LDESCRIPTION,
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
    packages=find_packages(),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    zip_safe=False
)
