Interpolation (:mod:`scipy.interpolate`)
========================================

.. sectionauthor:: Travis E. Oliphant

.. currentmodule:: scipy.interpolate

.. contents::

There are two general interpolation facilities available in SciPy. The
first facility is an interpolation class which performs linear
1-dimensional interpolation. The second facility is based on the
FORTRAN library FITPACK and provides functions for 1- and
2-dimensional (smoothed) cubic-spline interpolation.


Linear 1-d interpolation (:class:`interp1d`)
--------------------------------------------

The interp1d class in scipy.interpolate is a convenient method to
create a function based on fixed data points which can be evaluated
anywhere within the domain defined by the given data using linear
interpolation. An instance of this class is created by passing the 1-d
vectors comprising the data. The instance of this class defines a
:meth:`__call__ <interp1d.__call__>` method and can therefore by
treated like a function which interpolates between known data values
to obtain unknown values (it also has a docstring for help). Behavior
at the boundary can be specified at instantiation time. The following
example demonstrates it's use.

.. plot::

   >>> import numpy as np
   >>> from scipy import interpolate

   >>> x = np.arange(0,10)
   >>> y = np.exp(-x/3.0)
   >>> f = interpolate.interp1d(x, y)

   >>> xnew = np.arange(0,9,0.1)
   >>> import matplotlib.pyplot as plt
   >>> plt.plot(x,y,'o',xnew,f(xnew),'-')

..   :caption: One-dimensional interpolation using the
..             class :obj:`interpolate.interp1d`


Spline interpolation in 1-d (interpolate.splXXX)
------------------------------------------------

Spline interpolation requires two essential steps: (1) a spline
representation of the curve is computed, and (2) the spline is
evaluated at the desired points. In order to find the spline
representation, there are two different was to represent a curve and
obtain (smoothing) spline coefficients: directly and parametrically.
The direct method finds the spline representation of a curve in a two-
dimensional plane using the function :obj:`splrep`. The
first two arguments are the only ones required, and these provide the
:math:`x` and :math:`y` components of the curve. The normal output is
a 3-tuple, :math:`\left(t,c,k\right)` , containing the knot-points,
:math:`t` , the coefficients :math:`c` and the order :math:`k` of the
spline. The default spline order is cubic, but this can be changed
with the input keyword, *k.*

For curves in :math:`N` -dimensional space the function
:obj:`splprep` allows defining the curve
parametrically. For this function only 1 input argument is
required. This input is a list of :math:`N` -arrays representing the
curve in :math:`N` -dimensional space. The length of each array is the
number of curve points, and each array provides one component of the
:math:`N` -dimensional data point. The parameter variable is given
with the keword argument, *u,* which defaults to an equally-spaced
monotonic sequence between :math:`0` and :math:`1` . The default
output consists of two objects: a 3-tuple, :math:`\left(t,c,k\right)`
, containing the spline representation and the parameter variable
:math:`u.`

The keyword argument, *s* , is used to specify the amount of smoothing
to perform during the spline fit. The default value of :math:`s` is
:math:`s=m-\sqrt{2m}` where :math:`m` is the number of data-points
being fit. Therefore, **if no smoothing is desired a value of**
:math:`\mathbf{s}=0` **should be passed to the routines.**

Once the spline representation of the data has been determined,
functions are available for evaluating the spline
(:func:`splev`) and its derivatives
(:func:`splev`, :func:`splade`) at any point
and the integral of the spline between any two points (
:func:`splint`). In addition, for cubic splines ( :math:`k=3`
) with 8 or more knots, the roots of the spline can be estimated (
:func:`sproot`). These functions are demonstrated in the
example that follows.

.. plot::

   >>> import numpy as np
   >>> import matplotlib.pyplot as plt
   >>> from scipy import interpolate

   Cubic-spline

   >>> x = np.arange(0,2*np.pi+np.pi/4,2*np.pi/8)
   >>> y = np.sin(x)
   >>> tck = interpolate.splrep(x,y,s=0)
   >>> xnew = np.arange(0,2*np.pi,np.pi/50)
   >>> ynew = interpolate.splev(xnew,tck,der=0)

   >>> plt.figure()
   >>> plt.plot(x,y,'x',xnew,ynew,xnew,np.sin(xnew),x,y,'b')
   >>> plt.legend(['Linear','Cubic Spline', 'True'])
   >>> plt.axis([-0.05,6.33,-1.05,1.05])
   >>> plt.title('Cubic-spline interpolation')
   >>> plt.show()

   Derivative of spline

   >>> yder = interpolate.splev(xnew,tck,der=1)
   >>> plt.figure()
   >>> plt.plot(xnew,yder,xnew,np.cos(xnew),'--')
   >>> plt.legend(['Cubic Spline', 'True'])
   >>> plt.axis([-0.05,6.33,-1.05,1.05])
   >>> plt.title('Derivative estimation from spline')
   >>> plt.show()

   Integral of spline

   >>> def integ(x,tck,constant=-1):
   >>>     x = np.atleast_1d(x)
   >>>     out = np.zeros(x.shape, dtype=x.dtype)
   >>>     for n in xrange(len(out)):
   >>>         out[n] = interpolate.splint(0,x[n],tck)
   >>>     out += constant
   >>>     return out
   >>>
   >>> yint = integ(xnew,tck)
   >>> plt.figure()
   >>> plt.plot(xnew,yint,xnew,-np.cos(xnew),'--')
   >>> plt.legend(['Cubic Spline', 'True'])
   >>> plt.axis([-0.05,6.33,-1.05,1.05])
   >>> plt.title('Integral estimation from spline')
   >>> plt.show()

   Roots of spline

   >>> print interpolate.sproot(tck)
   [ 0.      3.1416]

   Parametric spline

   >>> t = np.arange(0,1.1,.1)
   >>> x = np.sin(2*np.pi*t)
   >>> y = np.cos(2*np.pi*t)
   >>> tck,u = interpolate.splprep([x,y],s=0)
   >>> unew = np.arange(0,1.01,0.01)
   >>> out = interpolate.splev(unew,tck)
   >>> plt.figure()
   >>> plt.plot(x,y,'x',out[0],out[1],np.sin(2*np.pi*unew),np.cos(2*np.pi*unew),x,y,'b')
   >>> plt.legend(['Linear','Cubic Spline', 'True'])
   >>> plt.axis([-1.05,1.05,-1.05,1.05])
   >>> plt.title('Spline of parametrically-defined curve')
   >>> plt.show()


Two-dimensional spline representation (:func:`bisplrep`)
--------------------------------------------------------

For (smooth) spline-fitting to a two dimensional surface, the function
:func:`bisplrep` is available. This function takes as required inputs
the **1-D** arrays *x*, *y*, and *z* which represent points on the
surface :math:`z=f\left(x,y\right).` The default output is a list
:math:`\left[tx,ty,c,kx,ky\right]` whose entries represent
respectively, the components of the knot positions, the coefficients
of the spline, and the order of the spline in each coordinate. It is
convenient to hold this list in a single object, *tck,* so that it can
be passed easily to the function :obj:`bisplev`. The
keyword, *s* , can be used to change the amount of smoothing performed
on the data while determining the appropriate spline. The default
value is :math:`s=m-\sqrt{2m}` where :math:`m` is the number of data
points in the *x, y,* and *z* vectors. As a result, if no smoothing is
desired, then :math:`s=0` should be passed to
:obj:`bisplrep` .

To evaluate the two-dimensional spline and it's partial derivatives
(up to the order of the spline), the function
:obj:`bisplev` is required. This function takes as the
first two arguments **two 1-D arrays** whose cross-product specifies
the domain over which to evaluate the spline. The third argument is
the *tck* list returned from :obj:`bisplrep`. If desired,
the fourth and fifth arguments provide the orders of the partial
derivative in the :math:`x` and :math:`y` direction respectively.

It is important to note that two dimensional interpolation should not
be used to find the spline representation of images. The algorithm
used is not amenable to large numbers of input points. The signal
processing toolbox contains more appropriate algorithms for finding
the spline representation of an image. The two dimensional
interpolation commands are intended for use when interpolating a two
dimensional function as shown in the example that follows. This
example uses the :obj:`mgrid <numpy.mgrid>` command in SciPy which is
useful for defining a "mesh-grid "in many dimensions. (See also the
:obj:`ogrid <numpy.ogrid>` command if the full-mesh is not
needed). The number of output arguments and the number of dimensions
of each argument is determined by the number of indexing objects
passed in :obj:`mgrid <numpy.mgrid>`.

.. plot::

   >>> import numpy as np
   >>> from scipy import interpolate
   >>> import matplotlib.pyplot as plt

   Define function over sparse 20x20 grid

   >>> x,y = np.mgrid[-1:1:20j,-1:1:20j]
   >>> z = (x+y)*np.exp(-6.0*(x*x+y*y))

   >>> plt.figure()
   >>> plt.pcolor(x,y,z)
   >>> plt.colorbar()
   >>> plt.title("Sparsely sampled function.")
   >>> plt.show()

   Interpolate function over new 70x70 grid

   >>> xnew,ynew = np.mgrid[-1:1:70j,-1:1:70j]
   >>> tck = interpolate.bisplrep(x,y,z,s=0)
   >>> znew = interpolate.bisplev(xnew[:,0],ynew[0,:],tck)

   >>> plt.figure()
   >>> plt.pcolor(xnew,ynew,znew)
   >>> plt.colorbar()
   >>> plt.title("Interpolated function.")
   >>> plt.show()

..   :caption: Example of two-dimensional spline interpolation.

Using radial basis functions for smoothing/interpolation
---------------------------------------------------------

Radial basis functions can be used for smoothing/interpolating scattered
data in n-dimensions, but should be used with caution for extrapolation
outside of the observed data range.

1-d Example
^^^^^^^^^^^

This example compares the usage of the Rbf and UnivariateSpline classes
from the scipy.interpolate module.

.. plot::

    >>> import numpy as np
    >>> from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
    >>> import matplotlib.pyplot as plt

    >>> # setup data
    >>> x = np.linspace(0, 10, 9)
    >>> y = np.sin(x)
    >>> xi = np.linspace(0, 10, 101)

    >>> # use fitpack2 method
    >>> ius = InterpolatedUnivariateSpline(x, y)
    >>> yi = ius(xi)

    >>> plt.subplot(2, 1, 1)
    >>> plt.plot(x, y, 'bo')
    >>> plt.plot(xi, yi, 'g')
    >>> plt.plot(xi, np.sin(xi), 'r')
    >>> plt.title('Interpolation using univariate spline')

    >>> # use RBF method
    >>> rbf = Rbf(x, y)
    >>> fi = rbf(xi)

    >>> plt.subplot(2, 1, 2)
    >>> plt.plot(x, y, 'bo')
    >>> plt.plot(xi, yi, 'g')
    >>> plt.plot(xi, np.sin(xi), 'r')
    >>> plt.title('Interpolation using RBF - multiquadrics')
    >>> plt.show()

..   :caption: Example of one-dimensional RBF interpolation.

2-d Example
^^^^^^^^^^^

This example shows how to interpolate scattered 2d data.

.. plot::

    >>> import numpy as np
    >>> from scipy.interpolate import Rbf
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib import cm

    >>> # 2-d tests - setup scattered data
    >>> x = np.random.rand(100)*4.0-2.0
    >>> y = np.random.rand(100)*4.0-2.0
    >>> z = x*np.exp(-x**2-y**2)
    >>> ti = np.linspace(-2.0, 2.0, 100)
    >>> XI, YI = np.meshgrid(ti, ti)

    >>> # use RBF
    >>> rbf = Rbf(x, y, z, epsilon=2)
    >>> ZI = rbf(XI, YI)

    >>> # plot the result
    >>> n = plt.normalize(-2., 2.)
    >>> plt.subplot(1, 1, 1)
    >>> plt.pcolor(XI, YI, ZI, cmap=cm.jet)
    >>> plt.scatter(x, y, 100, z, cmap=cm.jet)
    >>> plt.title('RBF interpolation - multiquadrics')
    >>> plt.xlim(-2, 2)
    >>> plt.ylim(-2, 2)
    >>> plt.colorbar()

