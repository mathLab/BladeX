Welcome to BladeX's documentation!
===================================================

Description
^^^^^^^^^^^^

BladeX is a Python package for blade generation.


Installation
--------------------
BladeX requires numpy, scipy, matplotlib, and sphinx (for the documentation). They can be easily installed via pip. 


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
--------------------

We made some tutorial examples. Please refer to the official GitHub repository for the last updates. Here the list of the exported tutorials:

- `Tutorial 1 <tutorial1dmd.html>`_ - Here we show how to prepare a blade 2D sectional profile through generating legacy and custom foils.
- `Tutorial 2 <tutorial2advdmd.html>`_ - Here we show how to proceed with preparing a blade 2D sectional profile by performing several transformation operations on the generated foils.
- `Tutorial 3 <tutorial3mrdmd.html>`_ - Here we show how to prepare a blade 3D sectional profiles by applying all the transformations due to the radial distribution of the blade parameters.
- `Tutorial 4 <tutorial4cdmd.html>`_ - Here we show how to deform a blade and its parametric curves by using a parameter file.


Developer's Guide
--------------------

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
