#%%
from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np
from glob import glob
from lightcone import lightcone_utils
from powerbox import PowerBox, LogNormalPowerBox
import camb
from camb import model, initialpower
from scipy.interpolate import UnivariateSpline
from get_power import pk1d
import h5py as h5
#%%
class linear_lightcone(object):
    def __init__(self, camb_results, N, dx, n_k=501, k_piv=None, z_piv=None, seed=None, kmin_fct=5.0, kmax_fct=5.0, no_evol=False):
        '''
        NOTE: all length in Mpc
        '''
        self.kmax = np.pi/dx
        self.kmin = 2*np.pi/(N*dx)
        # the z_in will always be increasing
        hubble = camb_results.Params.h
        k_in, z_in, pk_in = camb_results.get_matter_power_spectrum(minkh=self.kmin/kmin_fct/hubble, maxkh=kmax_fct*self.kmax/hubble, npoints=n_k) # default unit for k in (Mpc/h)^-1 and for pk is (Mpc/h)^3
        # convert to Mpc^-1 and Mpc^3
        k_in = k_in*hubble
        pk_in = pk_in/hubble**3
        sigma8 = camb_results.get_sigma8() # in order of increasing time (decreasing redshift)
        # reorder, now in the same order as z_in
        D_in = sigma8[::-1]/sigma8[-1]
        z_in = np.array(z_in)
        if no_evol:
            if z_piv is None:
                D_in[:] = 1.0
            else:
                D_in[:] = np.exp( np.interp(np.log(z_piv+1), np.log(z_in+1), np.log(D_in)) )
        self.Dfunc = lambda z: np.exp( np.interp(np.log(z+1), np.log(z_in+1), np.log(D_in)) )
        if z_piv is None:
            self.D_piv = 1.0
        else:
            self.D_piv = self.Dfunc(z_piv)
        pk0 = pk_in[0]
        if k_piv is not None:
            self.win_k = np.exp(-k_in**2/2.0/k_piv**2)
        else:
            self.win_k = np.ones_like(k_in)
        self.pk_in = pk0
        self.k_in = k_in
        self.z_in = z_in
        self.pkfunc_loglog = UnivariateSpline(np.log10(k_in), np.log10(pk0*self.win_k), s=0, k=3, ext=1)
        self.boxlength = dx*N
        self.dx = dx
        self.N = N
        self.lnpb = LogNormalPowerBox(N=N, dim=3, pk = lambda k: 10**self.pkfunc_loglog(np.log10(k))*self.D_piv**2, boxlength=dx*N, seed=seed)
        self.delta0 = self.lnpb.delta_x()/self.D_piv
        self.z_snap = None
        self.lu_obj = None
        self.lightcone = None

    def set_snap(self, z_snap):
        self.z_snap = np.asarray(z_snap)
        self.D_snap = self.Dfunc(self.z_snap)
        self.fields = np.array([self.delta0*ii for ii in self.D_snap])
        print('Initialize light-cone')
        self.lu_obj = lightcone_utils(self.z_snap, self.fields, self.boxlength)

    def set_los(self, th, phi, x0, y0, z0, drc, dx, dy, nx, ny, parallel_los):
        '''
        NOTE: x0 and y0 in Mpc but dx and dy in unit of deg
        '''
        self.lu_obj.set_los(th, phi, x0, y0, z0, drc, dx, dy, nx, ny, parallel_los=parallel_los)
        self.rc_lc = self.lu_obj.rc_lc
        self.z_lc = self.lu_obj.z_lc
        self.dx_lc = dx
        self.dy_lc = dy

    def interp_lc(self, upgrade=2, kernel_para=None, kernel_perp=None):
        self.lu_obj.interp_lc(upgrade=upgrade, kernel_para=kernel_para, kernel_perp=kernel_perp)
        self.lightcone = self.lu_obj.fields_lc

    def get_field_2D(self, z_min=None, z_max=None):
        if z_min is None: z_min = self.z_lc.min()
        if z_max is None: z_max = self.z_lc.max()
        valid = (self.z_lc>=z_min) & (self.z_lc<=z_max)
        lc = self.lightcone[...,valid]
        rc = self.rc_lc[valid]
        z = self.z_lc[valid]
        print('Generate 2D field from z=%.3f to %.3f'%(z[0], z[-1]))
        self.field_2D = np.trapz(lc, rc, axis=-1)
        D_int = self.Dfunc(z)
        integrand = D_int**2/rc**2
        self.cl_theory = np.trapz(integrand, rc) * self.pk_in*self.win_k
        self.l_theory = self.k_in*rc.mean()
        self.lmax = self.kmax * rc.mean()
        self.lmin = self.kmin * rc.mean()
        self.z_field_2D = z
        self.rc_field_2D = rc
        return self.field_2D, (self.l_theory, self.cl_theory)

#%%
if __name__ == '__main__':
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.InitPower.set_params(ns=0.965)
    #Note non-linear corrections couples to smaller scales than you want
    z_camb = np.logspace(0, 2, 101) - 1
    z_camb[0] = 0.0
    z_camb = z_camb[::-1]
    pars.set_matter_power(redshifts=z_camb, kmax=10.0)
    
    #Linear spectra
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    
    
    k_all, _, pk_all = results.get_linear_matter_power_spectrum(hubble_units=False, k_hunit=False)
    
    
    seed = 1010
    z_piv = 5.0
    
    ll = linear_lightcone(results, 128, 3.0, seed=seed, z_piv=z_piv)
    
    z_range = [5.0, 15.0]
    ll.set_snap(np.arange(z_range[0], z_range[1]+1.0, 1.0))
    
    pk, k, _ = pk1d(ll.delta0, dx=ll.dx, dk_fct=1.0, get_var=False)
    
    plt.figure()
    plt.loglog(k_all, pk_all[0])
    plt.loglog(ll.k_in, ll.pk_in)
    plt.figure()
    plt.loglog(ll.k_in, ll.pk_in)
    plt.loglog(k, pk)
    for ii in range(0, len(ll.z_snap), 4):
        pk, k, _ = pk1d(ll.fields[ii], dx=ll.dx, dk_fct=1.0, get_var=False)
        plt.loglog(k, pk/ll.Dfunc(ll.z_snap[ii])**2, alpha=0.3)
    plt.show()
    
    
    drc = np.linspace(0, 1500, 1501)
    z0 = 6.0
    idx = -1
    dx = 0.005
    dy = 0.005
    nx = 64
    ny = 64
    with h5.File('/home/fd426/xray_ana/lc_params/r_min_0.45_seed_101.hdf5', 'r') as filein:
        ll.set_los(filein['th'][idx], filein['phi'][idx], filein['x0'][idx]*ll.boxlength, filein['y0'][idx]*ll.boxlength, z0, drc, dx, dy, nx, ny, parallel_los=False)
        print(filein['r_max'][idx])
    
    ll.interp_lc()
    ll.get_field_2D()
    
    cl, l, _ = pk1d(ll.field_2D, dx=np.deg2rad([ll.dx_lc, ll.dy_lc]), dk_fct=1.0)

    plt.figure()
    plt.imshow(ll.field_2D.T, origin='lower')
    
    plt.figure()
    plt.loglog(l, cl)
    ylim = plt.ylim()
    xlim = plt.xlim()
    plt.loglog(ll.l_theory, ll.cl_theory)
    plt.vlines(ll.lmax, *ylim, color='red', ls='dashed')
    plt.ylim(*ylim)
    plt.xlim(*xlim)
    plt.show()
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%

