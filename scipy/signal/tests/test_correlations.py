import numpy as np
from numpy.testing import *
from scipy.signal import correlate, xcorr1

class TestXcorr(TestCase):
    def test1(self):
        x = np.random.randn(5)
        y = np.random.randn(3)

        z_r = correlate(x, y, 'full')
        z = xcorr1(x, y)
        assert_array_almost_equal(z, z_r)

    def test1_complex(self):
        x = np.random.randn(5) + 1j * np.random.randn(5)
        y = np.random.randn(3) + 1j * np.random.randn(3)

        z_r = correlate(x, y.conj(), 'full')
        z = xcorr1(x, y)
        assert_array_almost_equal(z, z_r)

        z = xcorr1(y, x)
        assert_array_almost_equal(z, z_r.conj()[::-1])

    def test1_complex_lag(self):
        x = np.random.randn(5) + 1j * np.random.randn(5)
        y = np.random.randn(3) + 1j * np.random.randn(3)

        z_r = correlate(x, y.conj(), 'full')
        z = xcorr1(x, y)
        assert_array_almost_equal(z, z_r)

        z = xcorr1(x, y, -2, 4)
        assert_array_almost_equal(z, z_r)

        z = xcorr1(x, y, -1, 3)
        assert_array_almost_equal(z, z_r[1:-1])

        z = xcorr1(x, y, 0, 1)
        assert_array_almost_equal(z, z_r[2:4])

    def test1_revert(self):
        x = np.random.randn(5)
        y = np.random.randn(3)

        z_r = correlate(x, y, 'full')[::-1]
        z = xcorr1(y, x)
        assert_array_almost_equal(z, z_r)

    def test1_lag(self):
        x = np.random.randn(5)
        y = np.random.randn(3)

        z_r = correlate(x, y, 'full')
        z = xcorr1(x, y, minlag=-2, maxlag=4)
        assert_array_almost_equal(z, z_r)

        z = xcorr1(x, y, minlag=-1, maxlag=3)
        assert_array_almost_equal(z, z_r[1:-1])

        z = xcorr1(x, y, minlag=0, maxlag=1)
        assert_array_almost_equal(z, z_r[2:4])

    def test1_revert_lag(self):
        x = np.random.randn(5)
        y = np.random.randn(3)

        z_r = correlate(x, y, 'full')[::-1]
        z = xcorr1(y, x, minlag=-4, maxlag=2)
        assert_array_almost_equal(z, z_r)

        z = xcorr1(y, x, minlag=-1, maxlag=1)
        assert_array_almost_equal(z, z_r[3:-1])

        z = xcorr1(y, x, minlag=0, maxlag=1)
        assert_array_almost_equal(z, z_r[4:6])

    def test2(self):
        x = np.random.randn(2, 5)
        y = np.random.randn(2, 3)

        z = xcorr1(x, y)
        for i in range(x.shape[0]):
            z_r = correlate(x[i], y[i], 'full')
            assert_array_almost_equal(z[i], z_r)

    def test2_axis(self):
        x = np.random.randn(3, 5)
        y = np.random.randn(2, 5)

        z = xcorr1(x, y, axis=0)
        for i in range(x.shape[1]):
            z_r = correlate(x[:,i], y[:,i], 'full')
            assert_array_almost_equal(z[:,i], z_r)

        x = np.random.randn(5, 3)
        y = np.random.randn(5, 2)

        z = xcorr1(x, y, axis=1)
        for i in range(x.shape[0]):
            z_r = correlate(x[i], y[i], 'full')
            assert_array_almost_equal(z[i], z_r)

    def test2_revert(self):
        x = np.random.randn(2, 5)
        y = np.random.randn(2, 3)

        z = xcorr1(y, x)
        for i in range(x.shape[0]):
            z_r = correlate(x[i], y[i], 'full')
            assert_array_almost_equal(z[i], z_r[::-1])

    def test3(self):
        x = np.random.randn(4, 2, 5)
        y = np.random.randn(4, 2, 3)

        z = xcorr1(x, y)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                z_r = correlate(x[i][j], y[i][j], 'full')
                assert_array_almost_equal(z[i][j], z_r)

    def test3_revert(self):
        x = np.random.randn(4, 2, 5)
        y = np.random.randn(4, 2, 3)

        z = xcorr1(y, x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                z_r = correlate(x[i][j], y[i][j], 'full')
                assert_array_almost_equal(z[i][j], z_r[::-1])

if __name__ == "__main__":
    run_module_suite()
