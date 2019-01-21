from setuptools import setup


def readme():
    """
    This function just return the content of README.md
    """
    with open('README.md') as f:
        return f.read()


setup(
    name='bladex',
    version='0.1.0',
    description='BladeX.',
    long_description=readme(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    keywords='blade-generation propeller iges procal',
    url='https://github.com/mathLab/BladeX',
    author='Marco Tezzele, Mahmoud Gadalla',
    author_email='marcotez@gmail.com, gadalla.mah@gmail.com',
    license='MIT',
    packages=['bladex'],
    install_requires=[
        'numpy', 'scipy', 'matplotlib', 'Sphinx==1.4', 'sphinx_rtd_theme'
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
    include_package_data=True,
    zip_safe=False)
