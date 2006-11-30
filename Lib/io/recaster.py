# Author: Matthew Brett

"""
Recaster class for recasting numeric arrays
"""

from numpy import *

def sctype_attributes():
    ''' Return dictionary describing numpy scalar types '''
    d_dict = {}
    for sc_type in ('complex','float'):
        t_list = sctypes[sc_type]
        for T in t_list:
            F = finfo(T)
            dt = dtype(T)
            d_dict[T] = {
                'kind': dt.kind,
                'size': dt.itemsize,
                'max': F.max,
                'min': F.min}
    for T in sctypes['int']:
        dt = dtype(T)
        sz = dt.itemsize
        bits = sz*8-1
        end = 2**bits
        d_dict[T] = {
            'kind': dt.kind,
            'size': sz,
            'min': -end,
            'max': end-1
            }
    for T in sctypes['uint']:
        dt = dtype(T)
        sz = dt.itemsize
        bits = sz*8
        end = 2**bits
        d_dict[T] = {
            'kind': dt.kind,
            'size': sz,
            'min': 0,
            'max': end
        }
    return d_dict

class Recaster(object):
    ''' Class to recast arrays to one of acceptable scalar types

    Initialization specifies acceptable types (ATs)

    Implements downcast and recast method - returns array that may be
    of different storage type to the input array, where the new type
    is one of the ATs. Downcast forces return array to be same size or
    smaller than the input.  recast method will return a larger type
    if no smaller type will contain the data without loss of
    precision.

    At its simplest, the downcast method can reject arrays that
    are not in the list of ATs.
    '''

    _sctype_trans = {'complex': 'c', 'c': 'c',
                     'float': 'f', 'f': 'f',
                     'int': 'i', 'i': 'i',
                     'uint': 'u', 'u': 'u'}

    _sctype_attributes = sctype_attributes()

    def __init__(self, sctype_list=None, sctype_tols=None):
        ''' Set types for which we are attempting to downcast

        Input
        sctype_list  - list of acceptable scalar types
                     If None defaults to all system types
        sctype_tols  - dictionary key datatype, values rtol, tol
                     to specify tolerances for checking near equality in downcasting
        '''
        if sctype_list is None:
            sctype_list = self._sctype_attributes.keys()
        self.sctype_list = sctype_list
        self.sctype_tols = self.default_sctype_tols()
        if sctype_tols is not None:
            self.sctype_tols.update(sctype_tols)
        # Cache sctype sizes, 
        self.sized_sctypes = {}
        for k in ('c', 'f', 'i', 'u'):
            self.sized_sctypes[k] = self.sctypes_by_size(k)
        # All integer sizes
        self.ints_sized_sctypes = []
        for k, v in self.sized_sctypes.items():
            if k in ('u', 'i'):
                for e in v:
                    self.ints_sized_sctypes.append(e)
        if self.ints_sized_sctypes:
            self.ints_sized_sctypes.sort(lambda x, y: cmp(y[1], x[1]))
        # Capable types list
        self._capable_sctypes = {}
        for k in self._sctype_attributes:
            self._capable_sctypes[k] = self.get_capable_sctype(k)
    
    def default_sctype_tols(self):
        ''' Default allclose tolerance values for all dtypes '''
        t_dict = {}
        for sc_type in ('complex','float'):
            t_list = sctypes[sc_type]
            for T in t_list:
                dt = dtype(T)
                F = finfo(dt)
                t_dict[T] = {
                    'rtol': F.eps,
                    'atol': F.tiny}
        F = finfo(float64)
        for sc_type in ('int', 'uint'):
            t_list = sctypes[sc_type]
            for T in t_list:
                dt = dtype(T)
                t_dict[T] = {
                    'rtol': F.eps,
                    'atol': F.tiny}
        return t_dict

    def sctypes_by_size(self, sctype):
        ''' Returns storage size ordered list of entries of scalar type sctype

        Input
        sctype   - one of "complex" or "c", "float" or "f" ,
                  "int" or "i", "uint" or "u"
        '''
        try:
            sctype = self._sctype_trans[sctype]
        except KeyError:
            raise TypeError, 'Did not recognize sctype %s' % sctype
        D = []
        for t in self.sctype_list:
            dt = dtype(t)
            if dt.kind == sctype:
                D.append([t, dt.itemsize])
        D.sort(lambda x, y: cmp(y[1], x[1]))
        return D

    def capable_sctype(self, sct):
        ''' Return smallest type containing sct type without precision loss

        Value pulled fron dictionary cached from init - see
        get_capable_sctype method for algorithm
        '''
        try:
            return self._capable_sctypes[sct]
        except KeyError:
            return None
        
    def get_capable_sctype(self, sct):
        ''' Return smallest scalar type containing sct type without precision loss

        Input
        sct     - scalar type

        ID = input type. AT = acceptable type.  Return ID if ID is
        in ATs. Otherwise return smallest AT that is larger than or
        same size as ID.

        If the desired sctype is an integer, returns the smallest
        integer (int or uint) that can contain the range of the input
        integer type

        If there is no type that can contain sct without loss of
        precision, return None
        '''
        if sct in self.sctype_list:
            return sct
        out_t = None
        # Unsigned and signed integers
        # Precision loss defined by max min outside datatype range
        D = self._sctype_attributes[sct]
        if D['kind'] in ('u', 'i'):
            out_t = self.smallest_int_sctype(D['max'], D['min'])
        else:
            # Complex and float types
            # Precision loss defined by data size < sct
            sctypes = self.sized_sctypes[D['kind']]
            if not sctypes:
                return None
            dti = D['size']
            out_t = None
            for i, t in enumerate(sctypes):
                if t[1] >= dti:
                    out_t = t[0]
                else:
                    break
        return out_t

    def tols_from_sctype(self, sctype):
        ''' Return rtol and atol for sctype '''
        tols = self.sctype_tols[sctype]
        return tols['rtol'], tols['atol']
        
    def smallest_same_kind(self, arr):
        ''' Return arr maybe downcast to same kind, smaller storage

        If arr cannot be downcast within given tolerances, then:
        return arr if arr is in list of acceptable types, otherwise
        return None
        '''
        dtp = arr.dtype
        dti = dtp.itemsize
        sctypes = self.sized_sctypes[dtp.kind]
        sctypes = [t[0] for i, t in enumerate(sctypes) if t[1] < dti]
        return self.smallest_from_sctypes(arr, sctypes)

    def smallest_from_sctypes(self, arr, sctypes):
        ''' Returns array recast to smallest possible type from list

        Inputs
        arr        - array to recast
        sctypes    - list of scalar types to try

        Returns None if no recast is within tolerance
        '''
        dt = arr.dtype.type
        rtol, atol = self.tols_from_sctype(dt)
        ret_arr = arr
        for T in sctypes:
            test_arr = arr.astype(T)
            if allclose(test_arr, arr, rtol, atol):
                ret_arr = test_arr
                can_downcast = True
            else:
                break
        if ret_arr.dtype.type not in self.sctype_list:
            return None
        return ret_arr
        
    def smallest_int_sctype(self, mx, mn):
        ''' Return integer type with smallest storage containing mx and mn

        Inputs
        mx      - maximum value
        mn      - minumum value

        Returns None if no integer can contain this range
        '''
        sct = None
        for T, tsz in self.ints_sized_sctypes:
            t_dict = self._sctype_attributes[T]
            if t_dict['max'] >= mx and t_dict['min'] <= mn:
                if sct is None or tsz < sz:
                    sct = T
                    sz = tsz
        return sct

    def recast(self, arr):
        ''' Try arr downcast, upcast if necesary to get compatible type '''
        dt = arr.dtype.type
        ret_arr = self.downcast(arr)
        if ret_arr is not None:
            return ret_arr
        # Could not downcast, arr dtype not in known list
        # Try upcast to larger dtype of same kind
        udt = self.capable_dtype[dt]
        if udt is not None:
            return arr.astype(udt)
        # We are stuck for floats and complex now
        # Can try casting integers to floats
        if arr.dt.kind in ('i', 'u'):
            sctypes = self.sized_sctypes['f']
            arr = self.smallest_from_sctypes(arr, sctypes)
            if arr is not None:
                return arr
        raise ValueError, 'Could not recast array within precision'
        
    def downcast(self, arr):
        dtk = arr.dtype.kind
        if dtk == 'c':
            return self.downcast_complex(arr)
        elif dtk == 'f':
            return self.downcast_float(arr)
        elif dtk in ('u', 'i'):
            return self.downcast_integer(arr)
        else:
            raise TypeError, 'Do not recognize array kind %s' % dtk
            
    def downcast_complex(self, arr):
        # can we downcast to float?
        dt = arr.dtype
        dti = ceil(dt.itemsize / 2)
        sctypes = self.sized_sctypes['f']
        flts = [t[0] for i, t in enumerate(sctypes) if t[1] <= dti]
        test_arr = arr.astype(flts[0])
        rtol, atol = self.tols_from_sctype(dt.type)
        if allclose(arr, test_arr, rtol, atol):
            return self.downcast_float(test_arr)
        # try downcasting to another complex type
        return self.smallest_same_kind(arr)
    
    def downcast_float(self, arr):
        # Try integer
        test_arr = self.downcast_integer(arr)
        rtol, atol = self.tols_from_sctype(arr.dtype.type)
        if allclose(arr, test_arr, rtol, atol):
            return test_arr
        # Otherwise descend the float types
        return self.smallest_same_kind(arr)

    def downcast_integer(self, arr):
        ''' Downcasts arr to integer

        Returns None if range of arr cannot be contained in acceptable
        integer types
        '''
        mx = amax(arr)
        mn = amin(arr)
        idt = self.smallest_int_sctype(mx, mn)
        if idt:
            return arr.astype(idt)
        return None
