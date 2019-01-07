Welcome to BladeX's documentation!
===================================================

.. image:: _static/logo_bladex.png
   :height: 150px
   :width: 150 px
   :align: right

Description
^^^^^^^^^^^^

BladeX is a Python package for geometrical parametrization and bottom-up construction of propeller blades. It allows to generate and deform a blade based on the radial distribution of its parameters such as pitch, rake, skew, and the sectional foils' parameters such as chord and camber. The package is ideally suited for parametric simulations on large number of blade deformations. It provides an automated procedure for the CAD generation, hence reducing the time and effort required for modelling. The main scope of BladeX is to deal with propeller blades, however it can be flexible to be applied on further applications with analogous geometrical structures such as aircraft wings, turbomachinery, or wind turbine blades.


Dependencies and installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
BladeX requires numpy, scipy, matplotlib, sphinx (for the documentation), and nose (for the local test). They can be easily installed using pip.

BladeX is compatible with Python 2.7 and Python 3.6. Moreover, some of the modules require OCC to be installed for the .iges or .stl CAD generation. Please see below for instructions on how to satisfy the OCC requirements. You can also refer to `pythonocc.org <https://pythonocc.org>`_ or `github.com/tpaviot/pythonocc-core <https://github.com/tpaviot/pythonocc-core>`_ for further instructions. 

Python2.7 OCC installation:
::

  conda install -c conda-forge -c dlr-sc -c pythonocc -c oce pythonocc-core==0.18.1 python=2.7

Python3.6 OCC installation:
::

  conda install -c conda-forge -c dlr-sc -c pythonocc -c oce pythonocc-core==0.18.1 python=3.6


The `official distribution <https://github.com/mathLab/BladeX>`_ is on GitHub, and you can clone the repository using
::

    git clone https://github.com/mathLab/BladeX

To install the package just type:
::

    python setup.py install

To uninstall the package you have to rerun the installation and record the installed files in order to remove them:

::

    python setup.py install --record installed_files.txt
    cat installed_files.txt | xargs rm -rf


Tutorials
^^^^^^^^^^^^^^^^

We made some tutorial examples. Please refer to the official GitHub repository for the last updates. Here the list of the exported tutorials:

- `Tutorial 1 <tutorial1generatefoils.html>`_ - Here we show how to prepare a blade 2D sectional profile through generating legacy and custom foils.
- `Tutorial 2 <tutorial2transformfoils.html>`_ - Here we show how to proceed with preparing a blade 2D sectional profile by performing several transformation operations on the generated foils.
- `Tutorial 3 <tutorial3generateblade.html>`_ - Here we show how to prepare a blade 3D sectional profiles by applying all the transformations due to the radial distribution of the blade parameters.
- `Tutorial 4 <tutorial4deformblade.html>`_ - Here we show how to deform a blade and its parametric curves by using a parameter file.


Developer's Guide
^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   code
   contact
   contributing
   LICENSE



Indices and tables
^^^^^^^^^^^^^^^^^^^^^^^^

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
