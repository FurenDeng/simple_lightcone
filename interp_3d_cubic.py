import numpy as np
import numba
from numba.typed import List

@numba.njit(parallel=True)
def _interp3d_k3(f, dxout, dyout, dzout, fout, h, n):
    '''
    Cubit interpolation for 3D data on regular grid with periodic boundary condition\n
    Credit to https://github.com/dbstein/fast_interp/blob/master/fast_interp/\n
    f: (Nx, Ny, Nz) the function to be interpolated at\n
    dxout, dyout, dzout: (N,) array, the position to evaluate the interpolation\n
    NOTE: the dxout, dyout and dzout is w.r.t the position of f[0,0,0]\n
    fout: (N,) output array\n
    h: (3,) the step size for x, y and z\n
    n: (3,) the shape of f\n
    '''
    m = fout.shape[0]
    for mi in numba.prange(m):
        xx = dxout[mi]
        yy = dyout[mi]
        zz = dzout[mi]
        ix = int(xx//h[0])
        iy = int(yy//h[1])
        iz = int(zz//h[2])
        ratx = xx/h[0] - (ix+0.5)
        raty = yy/h[1] - (iy+0.5)
        ratz = zz/h[2] - (iz+0.5)
        asx = np.empty(4)
        asy = np.empty(4)
        asz = np.empty(4)
        asx[0] = -1/16 + ratx*( 1/24 + ratx*( 1/4 - ratx/6))
        asx[1] =  9/16 + ratx*( -9/8 + ratx*(-1/4 + ratx/2))
        asx[2] =  9/16 + ratx*(  9/8 + ratx*(-1/4 - ratx/2))
        asx[3] = -1/16 + ratx*(-1/24 + ratx*( 1/4 + ratx/6))
        asy[0] = -1/16 + raty*( 1/24 + raty*( 1/4 - raty/6))
        asy[1] =  9/16 + raty*( -9/8 + raty*(-1/4 + raty/2))
        asy[2] =  9/16 + raty*(  9/8 + raty*(-1/4 - raty/2))
        asy[3] = -1/16 + raty*(-1/24 + raty*( 1/4 + raty/6))
        asz[0] = -1/16 + ratz*( 1/24 + ratz*( 1/4 - ratz/6))
        asz[1] =  9/16 + ratz*( -9/8 + ratz*(-1/4 + ratz/2))
        asz[2] =  9/16 + ratz*(  9/8 + ratz*(-1/4 - ratz/2))
        asz[3] = -1/16 + ratz*(-1/24 + ratz*( 1/4 + ratz/6))
        ix -= 1
        iy -= 1
        iz -= 1
        fout[mi] = 0.0
        for i in range(4):
            ixi = (ix + i) % n[0]
            for j in range(4):
                iyj = (iy + j) % n[1]
                for k in range(4):
                    izk = (iz + k) % n[2]
                    fout[mi] += f[ixi,iyj,izk]*asx[i]*asy[j]*asz[k]

def interp3d_cubic(f, xout, d, x0=None):
    '''
    Wrapper for _interp3d_k3.\n
    Cubit interpolation for 3D data on regular grid with periodic boundary condition\n
    Credit to https://github.com/dbstein/fast_interp/blob/master/fast_interp/\n
    f: (Nx, Ny, Nz) the function to be interpolated at\n
    xout: (N1, N2, N3, ..., 3) the coordinate to interpolate\n
    d: scalar or (3,) the step size for x, y and z\n
    x0: None, scalar or (3,), the coordinate for f[0,0,0]\n
    return: (N1, N2, N3, ...) the interpolated values
    '''
    f = np.asarray(f)
    assert np.ndim(f) == 3
    n = List(f.shape)
    #xout = np.asfortranarray(xout)
    if x0 is not None:
        xout = xout - np.asarray(x0)
    assert xout.shape[-1] == 3
    if np.ndim(d) == 0:
        d = [d]*3
    assert len(d) == 3
    d = List([float(ii) for ii in d])
    shp = xout.shape[:-1]
    dx = xout[...,0].reshape(-1)
    dy = xout[...,1].reshape(-1)
    dz = xout[...,2].reshape(-1)
    out = np.zeros(dx.shape[0], dtype=f.dtype)
    _interp3d_k3(f, dx, dy, dz, out, d, n)
    return out.reshape(shp)

if __name__ == '__main__':
    x = np.linspace(-2, 2, 101)
    y = np.linspace(-2, 3, 201)
    z = np.linspace(-1, 2, 81)
    xm, ym, zm = np.meshgrid(x, y, z, indexing='ij')
    
    deg = 3
    c = np.random.rand(3, deg+1)
    def get_val(xi, yi, zi):
        f = 0
        xi = (xi-x[0])%((x[1]-x[0])*x.shape[0]) + x[0]
        yi = (yi-y[0])%((y[1]-y[0])*y.shape[0]) + y[0]
        zi = (zi-z[0])%((z[1]-z[0])*z.shape[0]) + z[0]
        f = f + np.polyval(c[0], xi)
        f = f + np.polyval(c[1], yi)
        f = f + np.polyval(c[2], zi)
        return f
    
    f0 = get_val(xm, ym, zm)
    x0 = [x[0], y[0], z[0]]
    d = [ii[1]-ii[0] for ii in [x, y, z]]
    x_test = np.random.rand(300, 200, 3)
    n_bound = 3
    for ii in range(3):
        x_test[...,ii] *= (f0.shape[ii]-2*n_bound)*d[ii]
        x_test[...,ii] += x0[ii] + d[ii]*n_bound
        x_test[...,ii] += d[ii]*f0.shape[ii]
    print(x_test[...,0].min(), x_test[...,0].max())
    print(x_test[...,1].min(), x_test[...,1].max())
    print(x_test[...,2].min(), x_test[...,2].max())
    f_test = get_val(x_test[...,0], x_test[...,1], x_test[...,2])
    f_itp = interp3d_cubic(f0, x_test, d, x0=x0)
    print(np.abs(f_test-f_itp).max())
    print('----------')
    for ii, jj in zip(np.random.choice(x_test.shape[0], 10), np.random.choice(x_test.shape[1], 10)):
        this_f_itp = interp3d_cubic(f0, x_test[ii,jj], d, x0=x0)
        print(np.abs(this_f_itp - f_itp[ii,jj]))
    
    import time
    for ii in range(5):
        interp3d_cubic(f0, x_test, d, x0=x0)
    t0 = time.time()
    for ii in range(300):
        interp3d_cubic(f0, x_test, d, x0=x0)
    print(time.time()-t0)
    


