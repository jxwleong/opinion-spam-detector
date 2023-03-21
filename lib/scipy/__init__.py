"""
SciPy: A scientific computing package for Python
================================================

Documentation is available in the docstrings and
online at https://docs.scipy.org.

Contents
--------
SciPy imports all the functions from the NumPy namespace, and in
addition provides:

Subpackages
-----------
Using any of these subpackages requires an explicit import. For example,
``import scipy.cluster``.

::

 cluster                      --- Vector Quantization / Kmeans
 datasets                     --- Dataset methods
 fft                          --- Discrete Fourier transforms
 fftpack                      --- Legacy discrete Fourier transforms
 integrate                    --- Integration routines
 interpolate                  --- Interpolation Tools
 io                           --- Data input and output
 linalg                       --- Linear algebra routines
 linalg.blas                  --- Wrappers to BLAS library
 linalg.lapack                --- Wrappers to LAPACK library
 misc                         --- Various utilities that don't have
                                  another home.
 ndimage                      --- N-D image package
 odr                          --- Orthogonal Distance Regression
 optimize                     --- Optimization Tools
 signal                       --- Signal Processing Tools
 signal.windows               --- Window functions
 sparse                       --- Sparse Matrices
 sparse.linalg                --- Sparse Linear Algebra
 sparse.linalg.dsolve         --- Linear Solvers
 sparse.linalg.dsolve.umfpack --- :Interface to the UMFPACK library:
                                  Conjugate Gradient Method (LOBPCG)
 sparse.linalg.eigen          --- Sparse Eigenvalue Solvers
 sparse.linalg.eigen.lobpcg   --- Locally Optimal Block Preconditioned
                                  Conjugate Gradient Method (LOBPCG)
 spatial                      --- Spatial data structures and algorithms
 special                      --- Special functions
 stats                        --- Statistical Functions

Utility tools
-------------
::

 test              --- Run scipy unittests
 show_config       --- Show scipy build configuration
 show_numpy_config --- Show numpy build configuration
 __version__       --- SciPy version string
 __numpy_version__ --- Numpy version string

"""


# start delvewheel patch
def _delvewheel_init_patch_1_3_1():
    import ctypes
    import os
    import platform
    import sys
    libs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scipy.libs'))
    is_pyinstaller = getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')
    is_conda_cpython = platform.python_implementation() == 'CPython' and (hasattr(ctypes.pythonapi, 'Anaconda_GetVersion') or 'packaged by conda-forge' in sys.version)
    if sys.version_info[:2] >= (3, 8) and not is_conda_cpython or sys.version_info[:2] >= (3, 10):
        if not is_pyinstaller or os.path.isdir(libs_dir):
            os.add_dll_directory(libs_dir)
    else:
        load_order_filepath = os.path.join(libs_dir, '.load-order-scipy-1.10.1')
        if not is_pyinstaller or os.path.isfile(load_order_filepath):
            with open(os.path.join(libs_dir, '.load-order-scipy-1.10.1')) as file:
                load_order = file.read().split()
            for lib in load_order:
                lib_path = os.path.join(os.path.join(libs_dir, lib))
                if not is_pyinstaller or os.path.isfile(lib_path):
                    ctypes.WinDLL(lib_path)


_delvewheel_init_patch_1_3_1()
del _delvewheel_init_patch_1_3_1
# end delvewheel patch



from numpy import show_config as show_numpy_config
if show_numpy_config is None:
    raise ImportError(
        "Cannot import SciPy when running from NumPy source directory.")
from numpy import __version__ as __numpy_version__

# Import numpy symbols to scipy name space (DEPRECATED)
from ._lib.deprecation import _deprecated
import numpy as np
_msg = ('scipy.{0} is deprecated and will be removed in SciPy 2.0.0, '
        'use numpy.{0} instead')

# deprecate callable objects from numpy, skipping classes and modules
import types as _types  # noqa: E402
for _key in np.__all__:
    if _key.startswith('_'):
        continue
    _fun = getattr(np, _key)
    if isinstance(_fun, _types.ModuleType):
        continue
    if callable(_fun) and not isinstance(_fun, type):
        _fun = _deprecated(_msg.format(_key))(_fun)
    globals()[_key] = _fun
del np, _types

from numpy.random import rand, randn
_msg = ('scipy.{0} is deprecated and will be removed in SciPy 2.0.0, '
        'use numpy.random.{0} instead')
rand = _deprecated(_msg.format('rand'))(rand)
randn = _deprecated(_msg.format('randn'))(randn)

# fft is especially problematic, so was removed in SciPy 1.6.0
from numpy.fft import ifft
ifft = _deprecated('scipy.ifft is deprecated and will be removed in SciPy '
                   '2.0.0, use scipy.fft.ifft instead')(ifft)

from numpy.lib import scimath  # noqa: E402
_msg = ('scipy.{0} is deprecated and will be removed in SciPy 2.0.0, '
        'use numpy.lib.scimath.{0} instead')
for _key in scimath.__all__:
    _fun = getattr(scimath, _key)
    if callable(_fun):
        _fun = _deprecated(_msg.format(_key))(_fun)
    globals()[_key] = _fun
del scimath
del _msg, _fun, _key, _deprecated

# We first need to detect if we're being called as part of the SciPy
# setup procedure itself in a reliable manner.
try:
    __SCIPY_SETUP__
except NameError:
    __SCIPY_SETUP__ = False


if __SCIPY_SETUP__:
    import sys
    sys.stderr.write('Running from SciPy source directory.\n')
    del sys
else:
    try:
        from scipy.__config__ import show as show_config
    except ImportError as e:
        msg = """Error importing SciPy: you cannot import SciPy while
        being in scipy source directory; please exit the SciPy source
        tree first and relaunch your Python interpreter."""
        raise ImportError(msg) from e

    from scipy.version import version as __version__

    # Allow distributors to run custom init code
    from . import _distributor_init
    del _distributor_init

    from scipy._lib import _pep440
    # In maintenance branch, change to np_maxversion N+3 if numpy is at N
    # See setup.py for more details
    np_minversion = '1.19.5'
    np_maxversion = '1.27.0'
    if (_pep440.parse(__numpy_version__) < _pep440.Version(np_minversion) or
            _pep440.parse(__numpy_version__) >= _pep440.Version(np_maxversion)):
        import warnings
        warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
                      f" is required for this version of SciPy (detected "
                      f"version {__numpy_version__})",
                      UserWarning)
    del _pep440

    # This is the first import of an extension module within SciPy. If there's
    # a general issue with the install, such that extension modules are missing
    # or cannot be imported, this is where we'll get a failure - so give an
    # informative error message.
    try:
        from scipy._lib._ccallback import LowLevelCallable
    except ImportError as e:
        msg = "The `scipy` install you are using seems to be broken, " + \
              "(extension modules cannot be imported), " + \
              "please try reinstalling."
        raise ImportError(msg) from e

    from scipy._lib._testutils import PytestTester
    test = PytestTester(__name__)
    del PytestTester

    submodules = [
        'cluster',
        'datasets',
        'fft',
        'fftpack',
        'integrate',
        'interpolate',
        'io',
        'linalg',
        'misc',
        'ndimage',
        'odr',
        'optimize',
        'signal',
        'sparse',
        'spatial',
        'special',
        'stats'
    ]

    __all__ = submodules + [
        'LowLevelCallable',
        'test',
        'show_config',
        '__version__',
        '__numpy_version__'
    ]

    def __dir__():
        return __all__

    import importlib as _importlib

    def __getattr__(name):
        if name in submodules:
            return _importlib.import_module(f'scipy.{name}')
        else:
            try:
                return globals()[name]
            except KeyError:
                raise AttributeError(
                    f"Module 'scipy' has no attribute '{name}'"
                )