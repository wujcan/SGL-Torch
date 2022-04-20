import warnings
import traceback

try:
    from .style import *
except ImportError:
    traceback.print_exc()
    warnings.warn("Can not import 'reckit.plot'.")

del warnings
del traceback
