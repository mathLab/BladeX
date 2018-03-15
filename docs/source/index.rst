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
