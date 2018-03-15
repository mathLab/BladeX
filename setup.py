from setuptools import setup

def readme():
	"""
	This function just return the content of README.md
	"""
	with open('README.md') as f:
		return f.read()

setup(name='bladex',
	  version='0.0.1',
	  description='BladeX.',
	  long_description=readme(),
	  classifiers=[
	  	'Development Status :: 0 - Alpha',
	  	'License :: OSI Approved :: MIT License',
	  	'Programming Language :: Python :: 2.7',
	  	'Programming Language :: Python :: 3.6',
	  	'Intended Audience :: Science/Research',
	  	'Topic :: Scientific/Engineering :: Mathematics'
	  ],
	  keywords='blade',
	  url='https://github.com/mathLab/BladeX',
	  author='Marco Tezzele, Mahmoud Gadalla',
	  author_email='marcotez@gmail.com, mgadalla@sissa.it',
	  license='MIT',
	  packages=['bladex'],
	  install_requires=[
	  		'numpy',
	  		'scipy',
	  		'matplotlib',
	  		'Sphinx==1.4',
	  		'sphinx_rtd_theme'
	  ],
	  test_suite='nose.collector',
	  tests_require=['nose'],
	  include_package_data=True,
	  zip_safe=False)
