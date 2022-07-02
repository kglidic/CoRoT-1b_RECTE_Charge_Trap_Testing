# CoRoT-1b_RECTE_Charge_Trap_Testing

Atmospheric Characterization of Hot Jupiter CoRoT-1 b Using the Wide Field Camera 3 on the Hubble Space Telescope. Investigating CoRoT-1 b through its secondary eclipses and producing spectrophotometric light curves corrected for charge trapping, also known as the ramp effect in time-series observations with the WFC3. We found that, when correcting for the ramp effect and using the typically discarded first orbit, we are better capable of constraining and optimizing the emission and transmission spectra.

This repository contains files, scripts, and example Jupyter notebooks for spectra modeling from the Hubble Space Telescope secondary eclipse data on exoplanet CoRoT-1 b. The modeling methods laid out in this repository utlizes the [RECTE (Ramp Effect Charge Trapping Eliminator)](https://recte.readthedocs.io/en/latest/index.html) which models and corrects the systematic known as “ramp effect”. The contained scripts already contain a modified copy of RECTE. In particular, this repository contains a reproducible notebook on the full analysis of CoRoT-1 b thus far and a few notebooks that allow one to test how the ramp profiles are created. All required files to run these notebooks are contained here. 

# [Batman](https://www.cfa.harvard.edu/~lkreidberg/batman/index.html) 
a Python package for fast calculation of exoplanet transit light curves. 
The package supports calculation of light curves for any radially symmetric stellar limb darkening law, using a new integration algorithm for models that cannot be quickly 
calculated analytically.

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install batman.

```bash
pip install batman-package
```
# [Bokeh](https://docs.bokeh.org/en/latest/index.html)
Bokeh is an interactive visualization library for modern web browsers. 
It provides elegant, concise construction of versatile graphics, and affords high-performance interactivity over large or streaming datasets. 
Bokeh can help anyone who would like to quickly and easily make interactive plots, dashboards, and data applications.

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install bokeh.

```bash
pip install bokeh
```
# [Corner](https://corner.readthedocs.io/en/latest/install.html)
An illustrative representation of different projections of samples in high dimensional spaces.

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install corner.

```bash
pip install corner
```
# [Pandas](https://pandas.pydata.org/docs/#)
When working with tabular data, such as data stored in spreadsheets or databases, pandas is the right tool for you.
Pandas will help you to explore, clean and process your data.

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install pandas.

```bash
pip install pandas
```
# [Numpy](https://numpy.org/)
NumPy is the fundamental package for scientific computing in Python. 
It is a Python library that provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast 
operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, 
basic linear algebra, basic statistical operations, random simulation and much more.

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install numpy.

```bash
pip install numpy
```

# [Scipy](https://www.scipy.org/scipylib/index.html)
SciPy (pronounced “Sigh Pie”) is open-source software for mathematics, science, and engineering. 
The SciPy library depends on NumPy, which provides convenient and fast N-dimensional array manipulation. 
The SciPy library is built to work with NumPy arrays, and provides many user-friendly and efficient numerical routines such as routines for numerical 
integration and optimization.

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install scipy.

```bash
pip install scipy
```

# [tshirt](https://tshirt.readthedocs.io/en/latest/installation.html)
he Time Series Helper & Integration Reduction Tool tshirt is a general-purpose tool for time series science. Its main application is transiting exoplanet science.

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install tshirt.

```bash
pip install tshirt
```
```
