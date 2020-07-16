# sn_simulation

A set of python scripts used to simulate supernova light curves.

```
This software was developed within the LSST DESC using LSST DESC resources, and so meets the criteria given in, and is bound by, the LSST DESC Publication Policy for being a "DESC product".
We welcome requests to access code for non-DESC use; if you wish to use the code outside DESC please contact the developers.

```
## Release Status

|Release|Date|packages|
|---|---|---|
|v1.0.0|2020/07/15|sn_simu_wrapper_v1.0.0, sn_simulator_v1.0.0|

## Feedback, License etc

If you have comments, suggestions or questions, please [write us an issue](https://github.com/LSSTDESC/sn_simulation/issues).

This is open source software, available for re-use under the modified BSD license.

```
Copyright (c) 2020, the sn_simulation contributors on GitHub, https://github.com/LSSTDESC/sn_simulation/graphs/contributors.
All rights reserved.
```

## Content of sn_simulation ##
* **docs**: documentation for sphinx
* **\_\_init\_\_.py**
* **LICENCE**: licence file
* **README.md**: this readme
* **setup.py**: setup file for pip installation
* [**sn_simulator**](doc_package/sn_simulator.md): set of simulators
* [**sn_simu_wrapper**](doc_package/sn_simu_wrapper.md): set of wrappers to run the simulation
* **tests**: unit tests


## Complete tree ##
```bash
|-- LICENCE
|-- README.md
|-- __init__.py
|-- doc_package
|   |-- sn_simu_wrapper.md
|   |-- sn_simulator.md
|-- docs
|   |-- Makefile
|   |-- api
|   |   |-- sn_simu_wrapper.rst
|   |   |-- sn_simu_wrapper.sn_object.rst
|   |   |-- sn_simu_wrapper.sn_simu.rst
|   |   |-- sn_simulator.rst
|   |   |-- sn_simulator.sn_cosmo.rst
|   |   |-- sn_simulator.sn_fast.rst
|   |-- conf.py
|   |-- index.rst
|   |-- make.bat
|-- setup.py
|-- sn_simu_wrapper
|   |-- __init__.py
|   |-- sn_object.py
|   |-- sn_simu.py
|   |-- version.py
|-- sn_simulator
|   |-- __init__.py
|   |-- sn_cosmo.py
|   |-- sn_fast.py
|   |-- version.py
|-- tests
|   |-- testSNsimulation.py
|-- version.py
```