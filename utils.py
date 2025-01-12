#%%
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from glob import glob
import os
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid as cumtrapz
from glob import glob
from tqdm import tqdm
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator
import warnings
from scipy import fft
#%%
class cosmo_utils(object):
    def __init__(self, params_file=None, cosmo_params=None, **itp_table_kwargs):
        '''
        In consistent with 21cmSPACE\n
        All comoving distance in Mpc\n
        NOTE: this class assume flat Universe\n
        '''
        if params_file is None and cosmo_params is None:
            try:
                params_file = '%s/Planck_parameters.mat'%os.environ['PATH_21CMSPACE']
            except:
                raise Exception('Cannot find parameters for cosmology, input params_file or set environment variable PATH_21CMSPACE')
        if params_file is not None:
            print('Load parameters from %s'%params_file)
            cosmo_params = loadmat(params_file)
            self.H0 = cosmo_params['H0'][0][0]
            self.Om = cosmo_params['Om'][0][0]
            self.OLambda = cosmo_params['OLambda'][0][0]
            self.Ob = cosmo_params['Ob'][0][0]

            # for consistency with 21cmSPACE
            self.c = cosmo_params['c'][0][0] # in km/s
            self.Or = 8.5522e-05
        else:
            self.H0 = cosmo_params['H0']
            self.Om = cosmo_params['Om']
            self.OLambda = cosmo_params.get('OLambda', 1-self.Om)
            self.Ob = cosmo_params.get('Ob', 0.0)
            self.Or = cosmo_params.get('Or', 0.0)
            from astropy import constants as const
            self.c = cosmo_params.get('c', const.c.to_value('km/s'))
        self._t0 = 977792.22168079 # 1/(km/s/Mpc) in Myr

        self.generate_itp_table(**itp_table_kwargs)

    def generate_itp_table(self, z_min=5, z_max=100, nlnz=30001):
        lnzp1 = np.linspace(np.log(z_min+1), np.log(z_max+1), nlnz)
        zp1 = np.exp(lnzp1)
        drdlnzp1 = self.c/self.get_Hz(zp1-1) * zp1
        dtdlnzp1 = self._t0/self.get_Hz(zp1-1)
        delta_rc = cumtrapz(drdlnzp1, lnzp1, initial=0.0)
        delta_t = cumtrapz(dtdlnzp1, lnzp1, initial=0.0)
        self.lnzp1 = lnzp1
        self.delta_rc = delta_rc
        self.delta_t = delta_t # lookback time
        drcdz = lambda zin: self.c/self.get_Hz(zin)
        self.rc_min = quad(drcdz, 0, z_min)[0] # in Mpc
        dtdz = lambda zin: 1.0/self.get_Hz(zin)/(zin+1.0)
        self.tlb_min = self._t0 * quad(dtdz, 0, z_min)[0] # in Myr

    def drcdz(self, z):
        '''
        rc in Mpc\n
        '''
        z = np.asarray(z)
        return self.c/self.get_Hz(z)

    def dtdz(self, z):
        '''
        t is lookback time in Myr\n
        '''
        z = np.asarray(z)
        return self._t0/self.get_Hz(z)/(z+1.0)

    def drcdnu(self, z):
        '''
        rc in Mpc\n
        nu in MHz\n
        '''
        z = np.asarray(z)
        return -(1+z)**2/1420.0 * self.drcdz(z)

    def k2dnu(self, z, k):
        '''
        k in Mpc^-1\n
        nu in MHz
        '''
        z = np.asarray(z)
        k = np.asarray(k)
        drmax = 2*np.pi/k
        dnu = drmax/self.drcdnu(z)
        return np.abs(dnu)

    def get_rc(self, z):
        '''
        rc in Mpc\n
        '''
        lnzp1 = np.log(z+1)
        assert lnzp1.max() < self.lnzp1.max()
        drc = np.interp(lnzp1, self.lnzp1, self.delta_rc)
        return self.rc_min + drc

    def get_tlb(self, z):
        '''
        Lookback time in Myr\n
        '''
        lnzp1 = np.log(z+1)
        assert lnzp1.max() < self.lnzp1.max()
        dt = np.interp(lnzp1, self.lnzp1, self.delta_t) # Myr
        return self.tlb_min + dt

    def drc2z(self, z0, drc):
        '''
        drc in Mpc\n
        Convert from the comoving distance from z0 to the redshift\n
        '''
        lnz0p1 = np.log(z0+1)
        drc = np.asarray(drc)
        assert np.min(lnz0p1)>self.lnzp1[0] and np.max(lnz0p1)<self.lnzp1[-1]
        delta_r0 = np.interp(lnz0p1, self.lnzp1, self.delta_rc)
        this_drc = drc + delta_r0
        assert this_drc.min()>self.delta_rc[0] and this_drc.max()<self.delta_rc[-1]
        lnzp1_itp = np.interp(this_drc, self.delta_rc, self.lnzp1)
        zout = np.exp(lnzp1_itp)-1.0
        # for numerical consistency
        zout = np.asarray(zout)
        #zout[drc==0] = z0
        if np.ndim(z0) == 0:
            zout[drc==0] = z0
        else:
            z0, drc = np.broadcast_arrays(z0, drc)
            mask = drc==0
            zout[mask] = z0[mask]
        return zout

    def dt2z(self, z0, dt):
        '''
        dt lookback time in Myr\n
        Convert from the lookback time from z0 to the redshift\n
        '''
        lnz0p1 = np.log(z0+1)
        dt = np.asarray(dt)
        assert np.min(lnz0p1)>self.lnzp1[0] and np.max(lnz0p1)<self.lnzp1[-1]
        delta_t0 = np.interp(lnz0p1, self.lnzp1, self.delta_t)
        this_dt = dt + delta_t0
        assert this_dt.min()>self.delta_t[0] and this_dt.max()<self.delta_t[-1]
        lnzp1_itp = np.interp(this_dt, self.delta_t, self.lnzp1)
        zout = np.exp(lnzp1_itp)-1.0
        # for numerical consistency
        zout = np.asarray(zout)
        #zout[dt==0] = z0
        if np.ndim(z0) == 0:
            zout[dt==0] = z0
        else:
            z0, dt = np.broadcast_arrays(z0, dt)
            mask = dt==0
            zout[mask] = z0[mask]
        return zout

    def get_Hz(self, z):
        '''
        in km/s/Mpc\n
        '''
        return self.H0*np.sqrt(self.Om*(1+z)**3+self.OLambda+self.Or*(1+z)**4);

#%%
###### test for cosmo_utils
if __name__ == '__main__':
    cosmo = cosmo_utils()
    params = dict(H0=cosmo.H0, Om=cosmo.Om)
    from astropy.cosmology import FlatLambdaCDM
    cosmo0 = FlatLambdaCDM(H0=cosmo.H0, Om0=cosmo.Om, Tcmb0=0.0)
    cosmo1 = cosmo_utils(cosmo_params=params)

    z = np.linspace(9, 12, 101)
    rc = cosmo0.comoving_distance(z).to_value('Mpc')
    tlb = cosmo0.age(0).to_value('Myr')-cosmo0.age(z).to_value('Myr')
    z0 = 10
    rc0 = cosmo0.comoving_distance(z0).to_value('Mpc')
    tlb0 = cosmo0.age(0).to_value('Myr')-cosmo0.age(z0).to_value('Myr')
    print(quad(cosmo1.drcdz, 0, np.exp(cosmo1.lnzp1[0])-1)[0]/cosmo1.rc_min-1)
    print(quad(cosmo1.dtdz, 0, np.exp(cosmo1.lnzp1[0])-1)[0]/cosmo1.tlb_min-1)
    dtlb = tlb - tlb0
    drc = rc - rc0
    z_test = cosmo.drc2z(z0, drc)
    z1_test = cosmo1.drc2z(z0, drc)
    zt_test = cosmo.dt2z(z0, dtlb)
    zt1_test = cosmo1.dt2z(z0, dtlb)


    z_test2 = cosmo.drc2z(np.array([z0, z0+1]), drc[:,None])
    zt_test2 = cosmo.dt2z(np.array([z0, z0+1]), dtlb[:,None])
    print(np.abs(z_test2[:,0]-z_test).max())
    print(np.abs(zt_test2[:,0]-zt_test).max())
    print(np.abs(z_test2[:,1]-cosmo.drc2z(z0+1, drc)).max())
    print(np.abs(zt_test2[:,1]-cosmo.dt2z(z0+1, dtlb)).max())


    plt.figure()
    plt.plot(z, z_test/z-1, label='from rc')
    plt.plot(z, zt_test/z-1, label='from t')
    plt.legend()
    plt.figure()
    plt.plot(z, z1_test/z-1, label='from rc')
    plt.plot(z, zt1_test/z-1, label='from t')
    plt.legend()
    plt.figure(figsize=[10, 5])
    plt.subplot(121)
    plt.plot(rc, cosmo.get_rc(z)/rc-1)
    plt.subplot(122)
    plt.plot(tlb, cosmo.get_tlb(z)/tlb-1)
    plt.figure(figsize=[10, 5])
    plt.subplot(121)
    plt.plot(z, cosmo.get_rc(z)-rc)
    plt.subplot(122)
    plt.plot(z, cosmo.get_tlb(z)-tlb)
    plt.figure(figsize=[10, 5])
    plt.subplot(121)
    plt.plot(z, drc-(cosmo.get_rc(z)-cosmo.get_rc(z0)))
    plt.subplot(122)
    plt.plot(z, dtlb-(cosmo.get_tlb(z)-cosmo.get_tlb(z0)))

    print(cosmo.rc_min, cosmo0.comoving_distance(np.exp(cosmo.lnzp1[0])-1))
    plt.show()
#%%
def fft_upgrade_rfft(field, n):
    raise Exception('Using rfft to upgrade requires that the last entries of the fft to be zero, which is not prefered.')
    field = np.asarray(field)
    for ii in field.shape: assert ii%2==0
    if np.ndim(n) == 0:
        n = np.array([n]*np.ndim(field))
    else:
        n = np.asarray(n)
    assert np.all(n%2 == 0)
    field_fft = fft.rfftn(field)
    shp = np.array(list(field_fft.shape))
    shp[:-1] *= n[:-1]
    shp[-1] = (shp[-1]-1)*n[-1] + 1
    field_fft_high = np.zeros(shp, dtype=field_fft.dtype)
    sl = ()
    for ii in range(len(shp)-1):
        idx_mid = shp[ii]//2
        this_len = field_fft.shape[ii]
        sl +=(slice(idx_mid-this_len//2,idx_mid+this_len//2),)
    sl += (slice(0, field_fft.shape[-1]),)
    axes_shift = tuple(np.arange(len(shp)-1))
    field_fft_high[sl] = fft.fftshift(field_fft, axes=axes_shift)
    field_fft_high = fft.ifftshift(field_fft_high, axes=axes_shift)
    field_high = fft.irfftn(field_fft_high) * np.prod(n)
    return field_high, field_fft_high

def fft_upgrade(field, n):
    field = np.asarray(field)
    for ii in field.shape: assert ii%2==0
    if np.ndim(n) == 0:
        n = np.array([n]*np.ndim(field))
    else:
        n = np.asarray(n)
    assert np.all(n%2 == 0)
    field_fft = fft.fftn(field)
    shp = np.array(list(field_fft.shape)) * n
    field_fft_high = np.zeros(shp, dtype=field_fft.dtype)
    sl = ()
    for ii in range(len(shp)):
        idx_mid = shp[ii]//2
        this_len = field_fft.shape[ii]
        sl +=(slice(idx_mid-this_len//2,idx_mid+this_len//2),)
    field_fft_high[sl] = fft.fftshift(field_fft)
    field_fft_high = fft.ifftshift(field_fft_high) * np.prod(n)
    field_high = fft.ifftn(field_fft_high)
    if not np.issubdtype(field.dtype, np.complexfloating):
        field_high = field_high.real
    return field_high, field_fft_high

#%%
if __name__ == '__main__':
    np.random.seed(101)
    y = np.random.rand(128) + np.random.rand(128)*1.J

    #y = fft.rfft(y)
    #y[-1] = 0.0
    n_fine = 8
    y_new, _ = fft_upgrade(y, n_fine)
    print(np.abs(y_new[::n_fine]-y).max())
    plt.figure()
    plt.plot(y.real, '.')
    plt.plot(np.arange(y_new.shape[0])/n_fine, y_new.real)
    plt.figure()
    plt.plot(y.imag, '.')
    plt.plot(np.arange(y_new.shape[0])/n_fine, y_new.imag)
    plt.show()

    np.random.seed(101)
    nx, ny, nz = 16, 8, 32
    dx = 1./nx
    dy = 1./ny
    dz = 1./nz
    x = np.arange(nx)*dx
    y = np.arange(ny)*dy
    z = np.arange(nz)*dz
    kx = fft.fftfreq(nx, d=x[1]-x[0])*2*np.pi
    ky = fft.fftfreq(ny, d=y[1]-y[0])*2*np.pi
    kz = fft.rfftfreq(nz, d=z[1]-z[0])*2*np.pi
    field_fft = np.random.randn(kx.shape[0], ky.shape[0], kz.shape[0]) + 1.J*np.random.randn(kx.shape[0], ky.shape[0], kz.shape[0])
    field_fft[0,:,:] = 0.0
    field_fft[nx//2,:,:] = 0.0
    field_fft[:,ny//2,:] = 0.0
    field_fft[:,:,nz//2] = 0.0

    kxm, kym, kzm = np.meshgrid(kx, ky, kz, indexing='ij')
    kall = np.sqrt(kxm**2 + kym**2 + kzm**2)

    pk = np.zeros_like(kall)
    pk[kall!=0] = kall[kall!=0]**-2

    field_fft = field_fft*np.sqrt(pk)
    field = fft.irfftn(field_fft)
    #n = 4
    n = [16, 2, 4]
    field_high, field_fft_high = fft_upgrade(field, n)
    print(np.abs(field_high[::n[0],::n[1],::n[2]]-field).max())
    plt.figure(figsize=[15, 5])
    plt.subplot(121)
    plt.imshow(field[0].T, origin='lower', extent=[-0.5, ny+0.5, -0.5, nz+0.5], aspect='auto')
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(field_high[0].T, origin='lower', extent=[-0.5, ny+0.5, -0.5, nz+0.5], aspect='auto')
    plt.colorbar()
    plt.figure(figsize=[10, 5])
    plt.subplot(121)
    plt.imshow(field[:,0].T, origin='lower', extent=[-0.5, nx+0.5, -0.5, nz+0.5])
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(field_high[:,0].T, origin='lower', extent=[-0.5, nx+0.5, -0.5, nz+0.5])
    plt.colorbar()
    plt.show()

    fft0 = fft.rfftn(field)
    fft1 = fft.rfftn(field_high)
    kxh = fft.fftfreq(field_high.shape[0], d=nx*dx/field_high.shape[0])*2*np.pi
    kyh = fft.fftfreq(field_high.shape[1], d=ny*dy/field_high.shape[1])*2*np.pi
    kzh = fft.rfftfreq(field_high.shape[2], d=nz*dz/field_high.shape[2])*2*np.pi
    dkx = kxh[1] - kxh[0]
    dky = kyh[1] - kyh[0]
    dkz = kzh[1] - kzh[0]
    vx = np.where((kxh<=kx.max()+dkx*0.01) & (kxh>=kx.min()-dkx*0.01))[0]
    vy = np.where((kyh<=ky.max()+dky*0.01) & (kyh>=ky.min()-dky*0.01))[0]
    vz = np.where((kzh<=kz.max()+dkz*0.01))[0]
    fft1 = fft1[vx[:,None,None],vy[None,:,None],vz[None,None,:]]
    if np.ndim(n)==0:
        fft1 = fft1/n**3
    else:
        fft1 = fft1/np.prod(n)

    print(np.abs(fft1-fft0).max())
#%%
def linear_interp_2p(x, x1, y1, x2, y2):
    dydx = (y2-y1)/(x2-x1)
    dx = x - x1
    return dydx*dx + y1

def linear_interp_idx(xout, x, check_sorted=True, side='left'):
    if check_sorted: assert np.all(np.diff(x)>0)
    n = x.shape[0]
    i0 = np.searchsorted(x, xout, side=side)
    i1 = i0 - 1
    i1 = np.clip(i1, 0, n-2)
    i0 = np.clip(i0, 1, n-1)
    return i0, i1

def linear_interp(xout, x, y, check_sorted=True):
    '''
    Interpolate along the first axis of y\n
    x should be sorted in ascending order\n
    '''
    x = np.asarray(x)
    xout = np.asarray(xout)
    i0, i1 = linear_interp_idx(xout, x, check_sorted=check_sorted)
    y = np.asarray(y)
    return linear_interp_2p(xout, x[i0], y[i0].T, x[i1], y[i1].T).T
    #dydx = (y[i0] - y[i1]).T/(x[i0] - x[i1])
    #dxout = (xout - x[i1].T) # broadcast them
    #return (dxout*dydx).T + y[i1]
#%%
if __name__ == '__main__':

    x = np.linspace(0, 1, 51)
    c1 = np.random.rand(1, 5, 3) - 0.5
    c2 = np.random.rand(1, 5, 3) - 0.5
    y = c1.T*x + c2.T
    y = y.T
    xnew = np.linspace(0, 1, 101)
    ytest = c1.T*xnew + c2.T
    ytest = ytest.T
    ynew = linear_interp(xnew, x, y)
    i0, i1 = linear_interp_idx(xnew, x, side='right')
    ynew1 = linear_interp_2p(xnew, x[i0], y[i0].T, x[i1], y[i1].T).T
    print(y.shape, ytest.shape, ynew.shape)
    print(np.abs(ynew-ytest).max())
    print(np.abs(ynew1-ytest).max())


    x = np.linspace(0, 1, 51)
    c1 = np.random.rand(1, 5, 3) - 0.5
    c2 = np.random.rand(1, 5, 3) - 0.5
    y = c1.T*x**3 + c2.T*x**2
    y = y.T
    xnew = np.linspace(0, 1, 101)
    
    ynew = linear_interp(xnew, x, y)

    i0, i1 = linear_interp_idx(xnew, x, side='right')
    ynew1 = linear_interp_2p(xnew, x[i0], y[i0].T, x[i1], y[i1].T).T
    print(np.abs(ynew-ynew1).max())
    
    for i1, i2 in np.ndindex(y.shape[1:]):
        yitp = np.interp(xnew, x, y[:,i1, i2])
        print(np.abs(yitp-ynew[:,i1,i2]).max())
#%%
def extend_field(field, out=None):
    shp = [ii+1 for ii in field.shape]
    if out is None:
        out = np.zeros(shp, dtype=field.dtype)
    else:
        assert np.array_equal(out.shape, shp)
    sl = tuple(slice(None,-1) for _ in shp)
    out[sl] = field
    for ii in range(len(shp)):
        sl0 = tuple(0 if jj==ii else slice(None) for jj in range(len(shp)))
        sl1 = tuple(-1 if jj==ii else slice(None) for jj in range(len(shp)))
        out[sl1] = out[sl0]
    return out

if __name__ == '__main__':
    a = np.random.rand(5, 6, 4)
    b = extend_field(a)
    plt.figure()
    plt.subplot(121)
    plt.imshow(a[...,0])
    plt.subplot(122)
    plt.imshow(b[...,0])

    plt.figure()
    plt.subplot(121)
    plt.imshow(a[...,2])
    plt.subplot(122)
    plt.imshow(b[...,2])

    plt.figure()
    plt.subplot(121)
    plt.imshow(a[...,0])
    plt.subplot(122)
    plt.imshow(b[...,-1])

    plt.show()
#%%
def fft_smooth(field, dx, sigma):
    field = np.asarray(field)
    if np.ndim(dx) == 0:
        dx = [dx]*np.ndim(field)
    if np.ndim(sigma) == 0:
        sigma = [sigma]*np.ndim(field)
    freqall = [fft.fftfreq(field.shape[ii], d=dx[ii]) for ii in range(len(dx)-1)]
    freqall.append(fft.rfftfreq(field.shape[-1], d=dx[-1]))
    freqmesh = np.meshgrid(*freqall, indexing='ij')
    r2 = 0.0
    for sigmai, freqm in zip(sigma, freqmesh):
        r2 = r2 + freqm**2 * 2*np.pi**2 * sigmai**2
    fft = fft.rfftn(field)*np.exp(-r2)
    #win = np.exp(-freq2*2*np.pi**2*sigma**2)
    return fft.irfftn(fft)
#%%
if __name__ == '__main__':
    from scipy.ndimage import convolve
    f = np.zeros([32, 32], dtype=np.float64)
    f[20, 28] = 1.0
    sigma = [0.6, 0.5]
    dx = [0.32, 0.22]
    x = np.arange(np.around(10*sigma[0]/dx[0]))*dx[0]
    x = x - x.mean()
    y = np.arange(np.around(10*sigma[1]/dx[1]))*dx[1]
    y = y - y.mean()
    r2 = x[:,None]**2/sigma[0]**2 + y**2/sigma[1]**2
    win = np.exp(-r2/2.0)/(2*np.pi*np.prod(sigma))*np.prod(dx)
    print(win.shape)
    f_sm1 = fft_smooth(f, dx, sigma)
    f_sm2 = convolve(f, win, mode='wrap')
    #%%
    plt.figure()
    plt.imshow(f_sm1.T, origin='lower')
    plt.colorbar()
    plt.figure()
    plt.imshow(f_sm2.T, origin='lower')
    plt.colorbar()
    plt.figure()
    plt.imshow((f_sm1-f_sm2).T, origin='lower')
    plt.colorbar()
    plt.show()
#%%
def rebin_field(field, n_rebin, extra_dims=0):
    '''
    field: (m1, m2, m3, ..., n1, n2, n3, ...)\n
    the extra_dims is for m1, m2, ... and these dimensions will not be rebined\n
    n1, n2, n3 ... will be rebined by n_rebin\n
    n_rebin should be scalar or array_like with the same length as np.ndim(field) - extra_dims\n
    '''
    field = np.asarray(field)
    new_shp = list(field.shape[:extra_dims])
    if np.ndim(n_rebin) == 0:
        n_rebin = [n_rebin]*(np.ndim(field)-extra_dims)
    n_rebin = np.asarray(n_rebin, dtype=np.int64)
    assert len(n_rebin) == np.ndim(field)-extra_dims
    sl = [slice(None)]*extra_dims
    for n, ii in zip(n_rebin, field.shape[extra_dims:]):
        n_new = (ii//n)
        sl.append(slice(0, n_new*n))
        new_shp.append(n_new)
        new_shp.append(n)
    sl = tuple(sl)
    axis = np.arange(1, len(new_shp)-extra_dims, 2) + extra_dims
    axis = tuple(axis)
    field = field[sl]
    field = field.reshape(*new_shp).mean(axis=axis)
    return field

if __name__ == '__main__':
    a = np.random.randn(10, 20, 30, 51)
    a1 = a.reshape(2, 5, *a.shape[1:])
    a2 = a.reshape(2, 1, 5, *a.shape[1:])
    b = rebin_field(a, [3, 5, 6], extra_dims=1)
    b0 = rebin_field(a[0], [3, 5, 6], extra_dims=0)
    b1 = rebin_field(a1, [3, 5, 6], extra_dims=2)
    b2 = rebin_field(a2, [3, 5, 6], extra_dims=3)
    print(b.shape, b0.shape, b1.shape, b2.shape)
    c = a[:,:18,:,:48].reshape(10, 6, 3, 6, 5, 8, 6).mean(axis=(2, 4, 6))
    print(np.abs(b-c).max())
    print(np.abs(b[0]-b0).max())
    print(np.abs(b.reshape(a1.shape[:2]+b.shape[1:])-b1).max())
    print(np.abs(b.reshape(a2.shape[:3]+b.shape[1:])-b2).max())
#%%
def get_hist(field, bins, dx=1.0, sigma=None, n_rebin=None, regularize=False, density=True, **kwargs):
    if sigma is not None:
        field = fft_smooth(field, dx, sigma)
    if n_rebin is not None:
        field = rebin_field(field, n_rebin)
    if regularize:
        norm = (field**2).mean()
        field = field/np.sqrt(norm)
    hist, bins = np.histogram(field.reshape(-1), bins=bins, density=density, **kwargs)
    x = (bins[1:] + bins[:-1])/2.0
    return x, hist, bins
#%%
def digitize_1d(x, bin_edges):
    x = np.asarray(x)
    if np.ndim(x) == 0:
        scalar_x = True
        x = np.array([x])
    else:
        scalar_x = False
    idx = np.searchsorted(bin_edges, x, side='right')-1
    sel = idx==(len(bin_edges)-1)
    if np.any(sel):
        sel[sel] = x[sel]==bin_edges[-1]
        idx[sel] = len(bin_edges)-2
    if scalar_x: idx = idx[0]
    return idx

if __name__ == '__main__':
    x = np.random.randint(-1, 20, 40)
    bins = [0, 4, 8, 12]
    idx = digitize_1d(x, bins)
    for ii in range(len(bins)-1):
        this_x = x[idx==ii]
        v1 = this_x>=bins[ii]
        if ii==len(bins)-2:
            v2 = this_x<=bins[ii+1]
        else:
            v2 = this_x<bins[ii+1]
        print(np.all(v1), np.all(v2), np.all(v1&v2))
    print(np.unique(idx, return_counts=True))
    print(np.histogram(x, bins=bins)[0])

#%%
def trapz_step(x_mid, x1, y1, x2, y2):
    y_mid = linear_interp_2p(x_mid, x1, y1, x2, y2)
    t1 = (y1+y_mid)/2.0* (x_mid-x1)
    t2 = (y2+y_mid)/2.0* (x2-x_mid)
    return t1, t2

def piecewise_trapz(x, y, x_mid, x_min=None, x_max=None):
    '''
    NOTE: y can be multi-dimension and integrate along the last axis\n
    '''
    x = np.asarray(x)
    y = np.asarray(y)
    shp = list(y.shape)
    if x_min is not None:
        if x_min >= x[-1]: return 0.0, 0.0
        if x_min  <= x[0]:
            x_min = None
    if x_max is not None:
        if x_max <= x[0]: return 0.0, 0.0
        if x_max >= x[-1]:
            x_max = None
    if x_min is not None:
        idx = np.searchsorted(x, x_min, side='left')
        y_min = linear_interp([x_min], x, y.T).T
        y = y[...,idx:]
        x = x[idx:]
        x = np.append(x_min, x)
        y = np.append(y_min, y, axis=-1)
    if x_max is not None:
        idx = np.searchsorted(x, x_max, side='right')
        y_max = linear_interp([x_max], x, y.T).T
        y = y[...,:idx]
        x = x[:idx]
        x = np.append(x, x_max)
        y = np.append(y, y_max, axis=-1)
    if x_mid <= x[0]:
        return 0.0, np.trapz(y, x)
    elif x_mid >= x[-1]:
        return np.trapz(y, x), 0.0
    i2, i1 = linear_interp_idx(x_mid, x)
    t1 = np.trapz(y[...,:i2], x[:i2], axis=-1)
    t2 = np.trapz(y[...,i2:], x[i2:], axis=-1)
    e1, e2 = trapz_step(x_mid, x[i1], y[...,i1], x[i2], y[...,i2])
    return t1+e1, t2+e2

#%%
if __name__ == '__main__':
    x = np.linspace(-1, 3, 11)
    k = 0.53
    b = 2.423
    y = k*x + b
    x_min = -0.2
    x_max = 1.81
    x_mid = 1.42
    
    t1, t2 = piecewise_trapz(x, y, x_mid, x_min, x_max)
    x_min = max(x_min, x[0])
    x_max = min(x_max, x[-1])
    x_mid = max(x_min, min(x_mid, x_max))
    print(k/2.0*(x_max**2-x_min**2)+b*(x_max-x_min))
    print(t1, k/2.0*(x_mid**2-x_min**2)+b*(x_mid-x_min), k/2.0*(x_mid**2-x_min**2)+b*(x_mid-x_min)-t1)
    print(t2, k/2.0*(x_max**2-x_mid**2)+b*(x_max-x_mid), k/2.0*(x_max**2-x_mid**2)+b*(x_max-x_mid)-t2)

#%%
def histogram1d_discont(x, bins_lower, bins_upper, weights=None):
    '''
    Histogram with discontinuous bins\n
    Count or weighted sum in each [bins_lower[i], bins_upper[i]]\n
    '''
    x = np.asarray(x)
    bins_lower = np.asarray(bins_lower)
    bins_upper = np.asarray(bins_upper)
    assert bins_lower.shape[0] == bins_upper.shape[0]
    assert np.all(bins_upper>bins_lower)
    if weights is not None:
        weights = np.asarray(weights)
        assert weights.shape == x.shape
        weights = weights.reshape(-1)
    x = x.reshape(-1)
    if weights is None:
        x = np.sort(x)
    else:
        isort = np.argsort(x)
        x = x[isort]
        weights = weights[isort]
    idx_l = np.searchsorted(x, bins_lower, side='right')
    idx_u = np.searchsorted(x, bins_upper, side='right')
    invalid = (idx_l==x.shape[0])|(idx_u==0)
    idx_l[invalid] = -1 # mask it
    idx_u[invalid] = -1
    idx_l[idx_l>0] = idx_l[idx_l>0]-1
    if weights is None:
        out = idx_u - idx_l
    else:
        out = np.zeros(bins_upper.shape[0], dtype=weights.dtype)
        for ii, (il,iu) in enumerate(zip(idx_l, idx_u)):
            if il<0: continue
            out[ii] = weights[il:iu].sum()
    return out

def histogram1d(x, bins, weights=None):
    '''
    Wrapper for histogram1d_discont\n
    If bins is not 2D array, use numpy.histogram directly, otherwise use histogram1d_discont\n
    '''
    if np.ndim(bins) <= 1:
        x = np.asarray(x).reshape(-1)
        if weights is not None:
            weights = np.asarray(weights).reshape(-1)
        return np.histogram(x, bins=bins, weights=weights)
    elif np.ndim(bins) == 2:
        hist = histogram1d_discont(x, bins[0], bins[1], weights=weights)
        return hist, bins
    else:
        raise Exception('Only supper 1D bins (continuous histogram) or 2D bins (lower and upper bins for discontinuous histogram), but get %dD bins'%np.ndim(bins))

if __name__ == '__main__':
    x = np.linspace(0, 10, 100, endpoint=False)
    np.random.shuffle(x)
    x = x.reshape(2, 5, 10)
    lower = [1.0, 2.0, 5.0]
    upper = [3.0, 7.0, 6.0]
    weights = np.random.rand(*x.shape)
    #weights = None
    
    out = histogram1d(x, (lower, upper), weights=weights)[0]
    out2 = np.zeros_like(out)
    for ii in range(len(lower)):
        valid = (x>=lower[ii]) & (x<=upper[ii])
        if weights is None:
            out2[ii] = valid.sum()
        else:
            out2[ii] = weights[valid].sum()
    print(np.abs(out-out2).max()/out.std())
    
    x = np.random.randn(1000).reshape(10, 100)
    weights = np.random.rand(*x.shape)
    #weights = None
    #bins = 21
    bins = np.linspace(-1, 1, 101)
    
    hist, bins = histogram1d(x, bins=bins, weights=weights)
    x = x.reshape(-1)
    if weights is not None: weights=weights.reshape(-1)
    hist2, bins2 = np.histogram(x, bins=bins, weights=weights)
    print(np.abs(hist-hist2).max())
#%%
#%%
