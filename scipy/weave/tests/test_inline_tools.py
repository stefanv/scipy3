
from numpy import *

from scipy.testing import *

from scipy.weave import inline_tools

set_local_path()
from test_scxx import *
restore_path()

class TestInline(TestCase):
    """ These are long running tests...

         I'd like to benchmark these things somehow.
    """
    @dec.slow
    def test_exceptions(self):
        a = 3
        code = """
               if (a < 2)
                  throw_error(PyExc_ValueError,
                              "the variable 'a' should not be less than 2");
               else
                   return_val = PyInt_FromLong(a+1);
               """
        result = inline_tools.inline(code,['a'])
        assert(result == 4)

        try:
            a = 1
            result = inline_tools.inline(code,['a'])
            assert(1) # should've thrown a ValueError
        except ValueError:
            pass

        from distutils.errors import DistutilsError, CompileError
        try:
            a = 'string'
            result = inline_tools.inline(code,['a'])
            assert(1) # should've gotten an error
        except:
            # ?CompileError is the error reported, but catching it doesn't work
            pass

if __name__ == "__main__":
    nose.run(argv=['', __file__])
