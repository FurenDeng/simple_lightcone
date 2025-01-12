#%%
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import UnivariateSpline, LinearNDInterpolator, NearestNDInterpolator, interpn
import warnings
from scipy import fft
from utils import linear_interp, histogram1d
#%%
def plot_pn(x, y, err=None, c=None, ls_pos='-', ls_neg='--', label=None, n_interp=1000, **kwargs):
    '''
    NOTE: the line stype is determinted by the ending point
    '''
    x = np.asarray(x)
    y = np.asarray(y)
    invalid = np.isnan(y)
    y = y[~invalid]
    x = x[~invalid]
    if err is not None:
        err = np.asarray(err)[~invalid]
    eps = 1e-16*np.abs(y)[y!=0].min()
    sign = np.sign(y+eps) # No zero
    #print(sign)
    flip = np.where(np.diff(sign)!=0)[0]
    flip = np.append(flip+1, len(x))
    ist = 0
    for ied in flip:
        this_x = x[ist:ied]
        this_y = y[ist:ied]
        if ied != len(x):
            xnew = x[ied-1] + np.arange(1, n_interp)*(x[ied]-x[ied-1])/n_interp
            ynew = np.interp(xnew, x[ied-1:ied+1], y[ied-1:ied+1])
            icut = np.where(np.sign(ynew) != sign[ied-1])[0]
            if len(icut) == 0: # abs(y[ied]) is too small
                icut = len(xnew)
            else:
                icut = icut[0]
            xnew = xnew[:icut]
            ynew = ynew[:icut]
            this_x = np.concatenate([this_x, xnew])
            this_y = np.concatenate([this_y, ynew])
        if ist!=0:
            xnew = x[ist-1] + np.arange(1, n_interp)*(x[ist]-x[ist-1])/n_interp
            ynew = np.interp(xnew, x[ist-1:ist+1], y[ist-1:ist+1])
            icut = np.where(np.sign(ynew) == sign[ist])[0]
            if len(icut) == 0: # abs(y[ied]) is too small
                icut = 0
            else:
                icut = icut[0]
            xnew = xnew[icut:]
            ynew = ynew[icut:]
            this_x = np.concatenate([xnew, this_x])
            this_y = np.concatenate([ynew, this_y])
        if sign[ist] >= 0: 
            l = plt.errorbar(this_x, np.abs(this_y), c=c, ls=ls_pos, **kwargs)
        else:
            l = plt.errorbar(this_x, np.abs(this_y), c=c, ls=ls_neg, **kwargs)

        if c is None: c = l[0].get_color()
        ist = ied
    if err is not None:
        plt.errorbar(x, np.abs(y), err, linestyle='', color=c, **kwargs)
        if label is not None:
            plt.errorbar([], [], [], c=c, ls=ls_pos, label=label, **kwargs)
    elif label is not None:
        plt.errorbar([], [], c=c, ls=ls_pos, label=label, **kwargs)
    return c

#x = np.linspace(0, 10, 31)
#y = np.sin(1.3*x+np.random.rand())
#plt.plot(x, y)
#plot_pn(x, y, np.ones_like(y)*0.3, capsize=3.0, label='test1')
#plot_pn(x, y*3+0.5, label='test2')
#plot_pn(x, -x**0.3, label='test3')
#plot_pn(x, -x**0.3, 0.1*x**0.1, capsize=3.0, label='test4')
#plt.legend()
#plt.show()
##%%
def pk1d(field, dx, get_var=True, simple_var=True, field2=None, kbins=None, dk_fct=4.0, corr_func=lambda x, y: (x*y.conj()).real, win_norm=None):
    '''
    If field2 is provided, the number of dimension of field2 should not exceed field's.\n
    If get_var and simple_var, estimate variance simply by pk**2/(Nmode+1), where Nmode is the number of ceils in k space to estimate the pk for each k bin. You can calculate Nmode easily from this variance and the +1 is to correct the bias caused by variance in the estimated pk.\n
    NOTE: if win_norm is not None, the shrinkage of volume due to window would also be taken into account\n
    P(k) = corr_func(delta_1, delta2)/vol. If win_norm is None, use the volume of field as vol, otherwise use the volume of win_norm to estimate, i.e. sum(win_norm*dvol). We require the number of dimension is the same for win_norm and field. For example, for underlying field f and f2 and the measured ones are field=w*f and field2=w2*f2, then set win_norm=w*w2 would give an estimator of power spectrum between f and f2.\n
    NOTE: if provided win_norm would only be used to normalize the P(k) and would not be multiplied to fields inside the function, i.e. it has already be multiplied to the fields before input to the function\n
    NOTE: if field2 is None, win_norm should be the squre of the window function applied to field\n
    '''
    fft1 = fft.fftn(field)
    ndim = np.ndim(fft1)
    if np.ndim(dx) == 0:
        dx = [dx]*ndim
    fft1 = fft1 * np.prod(dx)
    if win_norm is None:
        vol = np.prod(dx)*np.size(field)
        vol_fct_var = 1.0
    else:
        win_norm = np.asarray(win_norm)
        assert np.ndim(win_norm) == ndim
        vol = np.sum(win_norm)*np.prod(dx)
        vol0 = np.prod(dx)*np.size(field)
        vol_fct_var = vol0*np.sum(win_norm**2)*np.prod(dx)/vol**2 # take into the effect of shrinkage in volume
    if field2 is None:
        fft2 = fft1
    else:
        ndim2 = np.ndim(field2)
        assert ndim2 <= ndim
        if ndim2 < ndim:
            fft2 = fft.fftn(field2)
            fft2 = np.expand_dims(fft2, axis=tuple(-np.arange(ndim-ndim2)-1))
        else:
            fft2 = fft.fftn(field2)
        fft2 = fft2*np.prod(dx[:ndim2])
        #fft2 = fft.fftn(field2)/np.size(field2)
    #corr = (fft1 * fft2.conj()).real
    #corr = corr * np.prod(dx) * np.size(field)
    corr = corr_func(fft1, fft2) / vol
    kall = [fft.fftfreq(field.shape[ii], d=dx[ii])*2*np.pi for ii in range(len(dx)-1)]
    kall.append(fft.fftfreq(field.shape[-1], d=dx[-1])*2*np.pi)
    kmesh = np.meshgrid(*kall, indexing='ij')
    k1d = 0.0
    for km in kmesh:
        k1d = k1d + km**2
    k1d = np.sqrt(k1d).reshape(-1)
    corr = corr.reshape(-1)
    if kbins is None:
        kmin = np.max([ii[1] for ii in kall])
        kmax = np.min([np.abs(ii).max() for ii in kall])
        #kmin = k1d[k1d!=0].min()
        dk = dk_fct*kmin
        #nk = np.around(k1d.max()/dk).astype(int)
        #kbins = np.geomspace(kmin/2.0, k1d.max()+dk/2.0, nk)
        #nk = np.around(kmax/dk).astype(int)
        #kbins = np.geomspace(kmin/2.0, kmax+kmin/2.0, nk)
        kbins = np.arange(0, kmax+kmin/2.0, dk)
        keep_length = False
    else:
        keep_length = True
    pk, kbins = histogram1d(k1d, bins=kbins, weights=corr)
    k, _ = histogram1d(k1d, bins=kbins, weights=k1d)
    norm, _ = histogram1d(k1d, bins=kbins)
    v = norm>0
    pk = pk[v]
    k = k[v]
    norm = norm[v]
    pk = pk/norm
    k = k/norm
    if get_var and simple_var:
        pkvar = pk**2*vol_fct_var/(1.0+norm/2.0)
    elif get_var:
        pkf = UnivariateSpline(k, pk, k=1, s=0, ext=0)
        pkitp = pkf(k1d)
        dpk = corr - pkitp
        pkvar, _ = histogram1d(k1d, bins=kbins, weights=dpk**2)
        pkvar = pkvar[v]
        pkvar = pkvar/norm
        pkvar[norm==1] = np.inf
        pkvar[norm>1] /= norm[norm>1]-1
    else:
        pkvar = None
    if keep_length:
        k_full = np.zeros(v.shape, k.dtype) + np.nan
        pk_full = np.zeros(v.shape, pk.dtype) + np.nan
        k_full[v] = k
        pk_full[v] = pk
        if pkvar is not None:
            pkvar_full = np.zeros(v.shape, pkvar.dtype) + np.nan
            pkvar_full[v] = pkvar
        else:
            pkvar_full = None
        return pk_full, k_full, pkvar_full
    else:
        return pk, k, pkvar
#%%
if __name__ == '__main__':
    from gather_data import xray_fields, Tb_HI_fields
    dirpath = '/home/fd426/rds/hpc-work/21cm/'
    prefix_single = 'fd_xray_IC_1040_single_popII'
    xf = xray_fields('%s/intermediate/xray_analysis/sim_%s/'%(dirpath, prefix_single), verbose=True)
    thf = Tb_HI_fields(xf.base_dirpath, verbose=True)
    
    z = 10
    f1 = xf.get_sfr_field(z)
    f2 = thf.get_Tb_field(z)
    f1 = f1 - f1.mean()
    f2 = f2 - f2.mean()
    pkx, kx, pkx_var = pk1d(f1, 3.0, field2=f2)
    pk1, k1, pk1_var = pk1d(f1, 3.0)
    pk2, k2, pk2_var = pk1d(f2, 3.0)
    plt.figure()
    plt.errorbar(k1, pk1, np.sqrt(pk1_var))
    plt.xscale('log')
    plt.yscale('log')
    plt.figure()
    plt.errorbar(k2, pk2, np.sqrt(pk2_var))
    plt.xscale('log')
    plt.yscale('log')
    plt.figure()
    plot_pn(kx, pkx, np.sqrt(pkx_var), capsize=3)
    plot_pn(k1, np.sqrt(pk1*pk2), np.sqrt((np.sqrt(pk1_var*pk2_var)+pkx_var)/2.0), capsize=3)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
#%%
def pk1d_multi_fields(fields, dx, fields2=None, remove_mean=True, k_out=None, win=None, win2=None, alpha_weight=2, interpolator=linear_interp, **kwargs):
    nf = len(fields)
    try:
        iter(fields2)
    except TypeError:
        fields2 = [fields2]*nf
    assert 'win_norm' not in kwargs, 'Do not input win_norm but input win, win2 and the window would be multiplied to fields and fields2 automatically.'
    if k_out is not None:
        k_out = np.asarray(k_out)
        k_shp = k_out.shape
        k_out = k_out.reshape(-1)
    first_iter = True
    for ii, (f1, f2) in enumerate(zip(fields, fields2)):
        if remove_mean:
            f1 = f1 - f1.mean()
            if f2 is not None: f2 = f2 - f2.mean()
        win_norm = 1
        if win is not None:
            f1 = f1 * win
            win_norm = win
        if f2 is None:
            if win_norm is not None: win_norm = win_norm**2
        elif win2 is not None:
            f2 = f2 * win2
            win_norm = win_norm * win2
        if np.array_equal(win_norm, 1): win_norm = None
        pk, k, _ = pk1d(f1, dx, field2=f2, win_norm=win_norm, **kwargs)
        if ii == 0:
            if k_out is None:
                pk_all = np.zeros([nf, k.shape[0]], dtype=pk.dtype)
            else:
                pk_all = np.zeros([nf, k_out.shape[0]], dtype=pk.dtype)
        if k_out is None:
            pk_all[ii] = pk
        else:
            y_itp = pk*k**alpha_weight
            pk_all[ii] = interpolator(np.log(k_out), np.log(k), y_itp)/k_out**alpha_weight
    if k_out is None:
        k_out = k
        k_shp = k.shape
    pk_all = pk_all.reshape((nf,)+k_shp)
    return pk_all, k_out

def pk2d(field, dx, get_var=True, simple_var=True, field2=None, kbins=None, dkp_fct=4.0, dkz_fct=4.0, corr_func=lambda x, y: (x*y.conj()).real, win_norm=None, broadcast_keff=True):
    '''
    If field2 is provided, the number of dimension of field2 should not exceed field's.\n
    If get_var and simple_var, estimate variance simply by pk**2/(Nmode+1), where Nmode is the number of ceils in k space to estimate the pk for each k bin. You can calculate Nmode easily from this variance and the +1 is to correct the bias caused by variance in the estimated pk.\n
    NOTE: if win_norm is not None, the shrinkage of volume due to window would also be taken into account\n
    P(k) = corr_func(delta_1, delta2)/vol. If win_norm is None, use the volume of field as vol, otherwise use the volume of win_norm to estimate, i.e. sum(win_norm*dvol). We require the number of dimension is the same for win_norm and field. For example, for underlying field f and f2 and the measured ones are field=w*f and field2=w2*f2, then set win_norm=w*w2 would give an estimator of power spectrum between f and f2.\n
    NOTE: if provided win_norm would only be used to normalize the P(k) and would not be multiplied to fields inside the function, i.e. it has already be multiplied to the fields before input to the function\n
    NOTE: if field2 is None, win_norm should be the squre of the window function applied to field\n
    '''
    fft1 = fft.fftn(field)
    ndim = np.ndim(fft1)
    if np.ndim(dx) == 0:
        dx = [dx]*ndim
    fft1 = fft1 * np.prod(dx)
    if win_norm is None:
        vol = np.prod(dx)*np.size(field)
        vol_fct_var = 1.0
    else:
        win_norm = np.asarray(win_norm)
        assert np.ndim(win_norm) == ndim
        vol = np.sum(win_norm)*np.prod(dx)
        vol0 = np.prod(dx)*np.size(field)
        vol_fct_var = vol0*np.sum(win_norm**2)*np.prod(dx)/vol**2 # take into the effect of shrinkage in volume
    if field2 is None:
        fft2 = fft1
    else:
        ndim2 = np.ndim(field2)
        assert ndim2 <= ndim
        if ndim2 < ndim:
            fft2 = fft.fftn(field2)
            fft2 = np.expand_dims(fft2, axis=tuple(-np.arange(ndim-ndim2)-1))
        else:
            fft2 = fft.fftn(field2)
        fft2 = fft2*np.prod(dx[:ndim2])
    #corr = (fft1 * fft2.conj()).real * vol
    corr = corr_func(fft1, fft2) / vol
    kall = [fft.fftfreq(corr.shape[ii], d=dx[ii])*2*np.pi for ii in range(len(dx)-1)]
    kall.append(fft.fftfreq(field.shape[-1], d=dx[-1])*2*np.pi)
    kmesh = np.meshgrid(*kall, indexing='ij')
    kzall = np.abs(kmesh[-1].reshape(-1))
    kpall = 0.0
    for km in kmesh[:-1]:
        kpall = kpall + km**2
    kpall = np.sqrt(kpall)
    kp1d = kpall[...,0].reshape(-1)
    kz1d = np.abs(kall[-1])
    kpall = kpall.reshape(-1)
    corr = corr.reshape(-1)
    #print(kzall.shape, corr.shape, kpall.shape)
    #if np.ndim(kbins) == 0: kbins = [kbins, kbins]
    try:
        iter(kbins)
    except TypeError:
        kbins = [kbins, kbins]
    if kbins[0] is None:
        #kmin = kpall[kpall!=0].min()
        #kmax = kpall[kpall!=0].max()
        kmin = np.max([ii[1] for ii in kall[:-1]])
        kmax = np.min([np.abs(ii).max() for ii in kall[:-1]])
        dk = dkp_fct*kmin
        #nk = np.around(kmax/dk).astype(int)
        #kbins[0] = np.linspace(-dk/2.0, kmax+dk/2.0, nk)
        kbins[0] = np.arange(0, kmax+kmin/2.0, dk)
    if kbins[1] is None:
        #kmin = kzall[kzall!=0].min()
        #kmax = kzall[kzall!=0].max()
        kmin = kall[-1][1]
        kmax = np.abs(kall[-1]).max()
        dk = dkz_fct*kmin
        #nk = np.around(kmax/dk).astype(int)
        #kbins[1] = np.linspace(-dk/2.0, kmax+dk/2.0, nk)
        #kbins[1] = np.arange(0, kmax+kmin/2.0, dk)
        kbins[1] = np.arange(-kmin/2.0, kmax+kmin/2.0, dk)
    pk, kpbins, kzbins = np.histogram2d(kpall, kzall, bins=kbins, weights=corr)

    norm_p, _ = np.histogram(kp1d, bins=kpbins)
    kpeff, _ = np.histogram(kp1d, bins=kpbins, weights=kp1d)
    v = norm_p>0
    kpeff[v] = kpeff[v]/norm_p[v]
    kpeff[~v] = np.nan

    norm_z, _ = np.histogram(kz1d, bins=kzbins)
    kzeff, _ = np.histogram(kz1d, bins=kzbins, weights=kz1d)
    v = norm_z>0
    kzeff[v] = kzeff[v]/norm_z[v]
    kzeff[~v] = np.nan

    if broadcast_keff:
        kpeff, kzeff = np.meshgrid(kpeff, kzeff, indexing='ij')

    kp = (kpbins[1:]+kpbins[:-1])/2.0
    kz = (kzbins[1:]+kzbins[:-1])/2.0

    norm = norm_p[:,None]*norm_z
    v = norm>0
    pk[v] = pk[v]/norm[v]
    pk[~v] = np.nan
    if get_var and simple_var:
        pkvar = pk**2*vol_fct_var/(1.0+norm/2.0)
    elif get_var:
        xitp = np.asfortranarray([kpeff[v], kzeff[v]]).T
        pkf = LinearNDInterpolator(xitp, pk[v])
        pkf1 = NearestNDInterpolator(xitp, pk[v])
        pkitp = pkf(kpall, kzall)
        iv1 = np.isnan(pkitp)
        pkitp[iv1] = pkf1(kpall[iv1], kzall[iv1])
        dpk = corr - pkitp
        pkvar, _, _ = np.histogram2d(kpall, kzall, bins=kbins, weights=dpk**2)
        pkvar[v] = pkvar[v]/norm[v]
        pkvar[~v] = np.nan
        pkvar[norm==1] = np.inf
        pkvar[norm>1] /= norm[norm>1]-1
    else:
        pkvar = None
    return pk, kp, kz, pkvar, (kpbins, kzbins), (kpeff, kzeff)

#%%
#%%
if __name__ == '__main__':
    from gather_data import xray_fields, Tb_HI_fields
    from matplotlib.colors import LogNorm
    z0 = 20.0
    dirpath = '/home/fd426/rds/hpc-work/21cm/'
    z_load = np.arange(-2, 3) + np.around(z0)
    prefix_single = 'fd_xray_IC_1040_single_popII'
    xf = xray_fields('%s/intermediate/xray_analysis/sim_%s/'%(dirpath, prefix_single), verbose=True, z=z_load)
    thf = Tb_HI_fields(xf.base_dirpath, verbose=True, z=z_load)

    f1 = xf.get_sfr_field(z0)
    f2 = thf.get_Tb_field(z0)
    f1 = f1 - f1.mean(axis=(0, 1), keepdims=True)
    f2 = f2 - f2.mean(axis=(0, 1), keepdims=True)
    #kbins = np.linspace(np.pi/(xf.Lpix*f1.shape[0]), np.pi/xf.Lpix, 21)
    kbins = None
    rc0 = xf.cosmo.get_rc(z0)
    pk1, kp1, kz1, pkvar1, (kpb, kzb), _ = pk2d(f1, [xf.Lpix, xf.Lpix, xf.Lpix], kbins=kbins)
    pk2, kp2, kz2, pkvar2, (kpb, kzb), _ = pk2d(f2, [xf.Lpix, xf.Lpix, xf.Lpix], kbins=kbins)
    #f2 = f2.mean(axis=-1)
    pkx, kpx, kzx, pkvarx, (kpb, kzb), (kp2d, kz2d) = pk2d(f1, [xf.Lpix, xf.Lpix, xf.Lpix], field2=f2, kbins=kbins)
    k2dlen = np.sqrt(kp2d**2 + kz2d**2)
    kbins1d = np.linspace(k2dlen.min()*0.99, k2dlen.max()*1.01, 21)
    pkx1d, kx1d, _ = pk1d(f1, [xf.Lpix, xf.Lpix, xf.Lpix], field2=f2, kbins=kbins1d)
    norm, _ = np.histogram(k2dlen.reshape(-1), bins=kbins1d)
    pkx2d21d, _ = np.histogram(k2dlen.reshape(-1), bins=kbins1d, weights=pkx.reshape(-1))
    pkx2d21d = pkx2d21d[norm>0]/norm[norm>0]

    plt.figure()
    plt.loglog(kx1d, pkx1d, '.')
    plt.loglog((kbins1d[1:]+kbins1d[:-1])/2.0, pkx1d)
    plt.show()

    pk1[0,0] = np.nan
    pk2[0,0] = np.nan
    pkx[0,0] = np.nan
    snr1 = np.abs(pk1)/np.sqrt(pkvar1)
    snr2 = np.abs(pk2)/np.sqrt(pkvar2)
    snrx = np.abs(pkx)/np.sqrt(pkvarx)

    plt.figure()
    plt.imshow(np.abs(pk1).T, origin='lower', extent=[kpb[0], kpb[-1], kzb[0], kzb[-1]], aspect='auto', interpolation='none', norm=LogNorm())
    plt.colorbar()
    plt.figure()
    plt.imshow(np.abs(pk2).T, origin='lower', extent=[kpb[0], kpb[-1], kzb[0], kzb[-1]], aspect='auto', interpolation='none', norm=LogNorm())
    plt.colorbar()
    plt.figure()
    plt.imshow(np.abs(pkx).T, origin='lower', extent=[kpb[0], kpb[-1], kzb[0], kzb[-1]], aspect='auto', interpolation='none', norm=LogNorm())
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(snr1.T, origin='lower', extent=[kpb[0], kpb[-1], kzb[0], kzb[-1]], aspect='auto', interpolation='none')
    plt.colorbar()
    plt.figure()
    plt.imshow(snr2.T, origin='lower', extent=[kpb[0], kpb[-1], kzb[0], kzb[-1]], aspect='auto', interpolation='none')
    plt.colorbar()
    plt.figure()
    plt.imshow(snrx.T, origin='lower', extent=[kpb[0], kpb[-1], kzb[0], kzb[-1]], aspect='auto', interpolation='none')
    plt.colorbar()
    plt.show()
#%%
def pdpk(field, n_sub, dx, field2=None, cal_2d=True, remove_mean_axis=None, **kwargs):
    field = np.asarray(field)
    ndim = np.ndim(field)
    if field2 is None:
        field2 = field
        ndim2 = ndim
    else:
        field2 = np.asarray(field2)
        ndim2 = np.ndim(field2)
        assert ndim2 <= ndim
        assert field2.shape == field.shape[:ndim2]
        if ndim2 < ndim:
            field2 = np.expand_dims(field2, axis=np.arange(-1, ndim2-ndim-1, -1).tolist())
    if np.ndim(n_sub) == 0:
        n_sub = [n_sub]*ndim2 + [1]*(ndim-ndim2)
    elif len(n_sub) == ndim2:
        n_sub = list(n_sub) + [1]*(ndim-ndim2)
    else:
        assert len(n_sub) == ndim
    sl1 = tuple()
    sl2 = tuple()
    n1 = []
    n2 = []
    for ii in range(ndim):
        n1.append(field.shape[ii]//n_sub[ii])
        n2.append(field2.shape[ii]//n_sub[ii])
        sl1 = sl1 + (slice(0, n1[-1] * n_sub[ii]),)
        sl2 = sl2 + (slice(0, n2[-1] * n_sub[ii]),)
    field = field[sl1]
    field2 = field2[sl2]
    pk_arr = []
    mean_arr = []
    pd = 0.0
    for idx in np.ndindex(tuple(n_sub)):
        idx1 = tuple()
        idx2 = tuple()
        for ii in range(ndim):
            ist1 = idx[ii]*n1[ii]
            idx1 = idx1 + (slice(ist1, ist1+n1[ii]),)
            ist2 = idx[ii]*n2[ii]
            idx2 = idx2 + (slice(ist2, ist2+n2[ii]),)
        f1 = field[idx1]
        f2 = field2[idx2]
        if remove_mean_axis == 'all':
            f1 = f1 - f1.mean()
        elif remove_mean_axis is not None:
            f1 = f1 - f1.mean(axis=remove_mean_axis, keepdims=True)
        mean_arr.append(f2.mean())
        if cal_2d:
            pk, kp, kz, _, kbins, keff = pk2d(f1, dx=dx, get_var=False, **kwargs)
            k_props = (kp, kz, kbins, keff)
        else:
            pk, k, _ = pk1d(f1, dx=dx, get_var=False, **kwargs)
            k_props = k
        pk_arr.append(pk)
    pk_arr = np.array(pk_arr)
    mean_arr = np.array(mean_arr)
    mean_arr = mean_arr - mean_arr.mean()
    pd = (pk_arr.T*mean_arr).mean(axis=-1).T
    return pd, (pk_arr, mean_arr), k_props
#%%
if __name__ == '__main__':
    from gather_data import xray_fields, Tb_HI_fields
    dirpath = '/home/fd426/rds/hpc-work/21cm/'
    prefix = 'fd_xray_IC_1040'
    dirname = '%s/intermediate/xray_analysis/sim_%s/'%(dirpath, prefix)
    z_test = [6, 7, 8, 9, 10]
    
    xfII = xray_fields(dirname, pop='II', verbose=True, z=z_test)
    xfIII = xray_fields(dirname, pop='III', verbose=True, z=z_test)
    #thf = Tb_HI_fields(xfII.base_dirpath, verbose=True, include_rsd=True, z=z_test)
    thf = Tb_HI_fields(xfII.base_dirpath, verbose=True, include_rsd=False, z=z_test)
    
    #n_sub = 32
    #kbins = None
    
    n_sub = 4
    #kmax = np.pi/xfII.Lpix
    #kmin = 2*np.pi/xfII.Lpix/(xfII.sfr_fields.shape[-1]//n_sub)
    #kbins = [np.arange(kmin/2.0, kmax-kmin/2.0, kmin*2.0)]*2
    kbins = None
    
    for idx_z in range(len(z_test)):
        sfr = xfII.sfr_fields[idx_z] + xfIII.sfr_fields[idx_z]
        Tb = thf.Tb_fields[idx_z]
        #Tb = Tb - Tb.mean(axis=(0,1), keepdims=True)
        #sfr = sfr - sfr.mean()
        plt.figure(figsize=[15, 4])
        ifig = 0
        for kind in ['full_3d', 'projected']:
            if kind == 'full_3d':
                f2 = sfr
            else:
                f2 = sfr.mean(axis=-1)
            pdpk_arr, (pk, mean), (_, _, (kpb, kzb), (kp, kz)) = pdpk(Tb, n_sub, xfII.Lpix, field2=f2, remove_mean_axis='all', kbins=kbins, dkp_fct=2.0, dkz_fct=2.0)
    
    
            extent = [kpb[0], kpb[-1], kzb[0], kzb[-1]]
            plt.subplot(131+ifig*2)
            #klen = np.sqrt(kp**2 + kz**2)
            #plt.imshow((pdpk_arr*klen**3/pk.mean(axis=0)/sfr.mean()).T, origin='lower', extent=extent)
            plt.imshow((pdpk_arr/pk.mean(axis=0)/sfr.mean()).T, origin='lower', extent=extent)
            plt.colorbar()
            plt.xlabel(r'$k_\perp/{\rm Mpc^{-1}}$')
            plt.ylabel(r'$k_\parallel/{\rm Mpc^{-1}}$')
    
            pdpk_arr, (pk, mean), k = pdpk(Tb, n_sub, xfII.Lpix, field2=f2, remove_mean_axis='all', kbins=kbins, dk_fct=2.0, cal_2d=False)
            plt.subplot(132)
            plt.semilogx(k, pdpk_arr/pk.mean(axis=0)/sfr.mean())
            #plt.xscale('log')
            #plt.yscale('log')
            plt.xlabel(r'$k/{\rm Mpc^{-1}}$')
            plt.ylabel(r'$k_\parallel/{\rm Mpc^{-1}}$')
            ifig += 1
        plt.title('z=%.1f, pdpk/pkHI/sfr_mean'%z_test[idx_z])
        plt.tight_layout()
        plt.show()
#%%
#%%
#%%
#%%
##%%
##inu = 100
#numin_keV = 0.5
#numax_keV = 8
#f1 = Tblc - Tblc.mean(axis=(0,1), keepdims=True)
##f2 = xrblc[...,inu]
#nu_sel = (xf.nu_keV>=numin_keV) & (xf.nu_keV<=numax_keV)
#f2 = np.trapz(xrblc[...,nu_sel], xf.nu_keV[nu_sel], axis=-1)
#print(xf.nu_keV[nu_sel][0], xf.nu_keV[nu_sel][-1])
#f2 = f2 - f2.mean()
##f2 = xf.sfr_lc
##f2 = f2 - f2.mean(axis=(0,1), keepdims=True)
##pkx, kpx, kzx, pkvarx, (kpb, kzb), _ = pk2d(f1, xf.Lpix, field2=f2, dkz_fct=3, dkp_fct=3)
#rc0 = xf.cosmo.get_rc(z0)
#pkx, kpx, kzx, pkvarx, (kpb, kzb), _ = pk2d(f1, [xf.Lpix/rc0, xf.Lpix/rc0, xf.Lpix], field2=f2, dkz_fct=3, dkp_fct=3)
#pkx[0,0] = np.nan
#Dkx = pkx*(kpx*(kpx+1))[:,None]/(2*np.pi)
##%%
#from matplotlib.colors import LogNorm
#plt.figure(dpi=150, figsize=[10, 3])
#plt.subplot(121)
#plt.imshow(np.abs(Dkx).T, origin='lower', norm=LogNorm(), extent=[kpb[0], kpb[-1], kzb[0], kzb[-1]], aspect='auto', interpolation='none');
#plt.xlabel(r'$k_\perp/{\rm Mpc^{-1}}$')
#plt.ylabel(r'$k_\parallel/{\rm Mpc^{-1}}$')
##plt.imshow(pkx.T, origin='lower');
#plt.colorbar()
#plt.title(r'$|D^{\rm HI\times XRB}(k_\perp, k_\parallel)|$')
#plt.subplot(122)
#plt.imshow(np.abs(pkx/np.sqrt(pkvarx)).T, origin='lower', extent=[kpb[0], kpb[-1], kzb[0], kzb[-1]], aspect='auto', interpolation='none');
#plt.xlabel(r'$k_\perp/{\rm Mpc^{-1}}$')
#plt.ylabel(r'$k_\parallel/{\rm Mpc^{-1}}$')
#plt.colorbar()
#plt.title('SNR')
#plt.figure(dpi=150)
#extent = [kpb[0], kpb[-1], xf.cosmo.k2dnu(z0, kzb[0]), xf.cosmo.k2dnu(z0, kzb[-1])]
#plt.imshow(np.abs(Dkx).T, origin='lower', norm=LogNorm(), extent=extent, aspect='auto', interpolation='none')
#plt.title(r'$|D^{\rm HI\times XRB}(k_\perp, k_\parallel)|$')
#plt.xlabel(r'$k_\perp/{\rm Mpc^{-1}}$')
#plt.ylabel(r'$\Delta\nu/MHz$')
#plt.colorbar()
#plt.figure()
#plt.imshow(pkx.T>0, origin='lower');
##%%
#plt.figure(figsize=[15, 3])
#plt.subplot(121)
##plot_pk(kpx, pkx[:,0]*(kpx+1)*kpx/(2*np.pi), np.sqrt(pkvarx[:,0])*(kpx+1)*kpx/(2*np.pi))
#plt.errorbar(kpx, pkx[:,0]*(kpx+1)*kpx/(2*np.pi), np.sqrt(pkvarx[:,0])*(kpx+1)*kpx/(2*np.pi))
#plt.xscale('log')
#plt.xlabel(r'$k_\perp/{\rm Mpc^{-1}}$')
#plt.ylabel(r'$|D^{\rm HI\times XRB}(k_\perp, k_\parallel)|$')
#plt.title('The first $k_\parallel$ bin')
#plt.subplot(122)
#plt.errorbar(kpx[1:], pkx[1:,1]*(kpx[1:]+1)*kpx[1:]/(2*np.pi), np.sqrt(pkvarx[1:,1])*(kpx[1:]+1)*kpx[1:]/(2*np.pi))
#plt.xlabel(r'$k_\perp/{\rm Mpc^{-1}}$')
#plt.ylabel(r'$D^{\rm HI\times XRB}(k_\perp, k_\parallel)$')
#plt.title('The second $k_\parallel$ bin')
##%%
#plt.figure()
#plt.errorbar(kzx[1:], pkx[1,1:], np.sqrt(pkvarx[1,1:]))
#plt.xlabel(r'$k_\parallel/{\rm Mpc^{-1}}$')
#plt.ylabel(r'$P^{\rm HI\times XRB}(k_\perp, k_\parallel)$')
##%%
##%%
##%%
#print(xf.cosmo.k2dnu(0.01/0.7, 1))
##%%
##%%
##%%
#z0 = 10.0
#xrblc = xf.get_lc(z0)
#Tblc = thf.get_lc(z0)
##%%
#inu = 100
#Tbcut = fft.rfft(Tblc - Tblc.mean(axis=(0,1), keepdims=True), axis=-1)
#kbcut = 2*np.pi*fft.rfftfreq(Tblc.shape[-1], d=xf.Lpix)
#dnucut = xf.cosmo.k2dnu(z0, kbcut)
#icut = np.where(dnucut < 1)[0][0]
#kmin = kbcut[icut]
#Tbcut[...,:icut] = 0.0
#print(kmin, icut, dnucut[icut])
#Tbcut = fft.irfft(Tbcut, axis=-1)
##%%
#print(xf.nu_keV[inu])
##f1 = Tbcut**2
#f1 = Tbcut
#f1 = f1 - f1.mean(axis=(0,1), keepdims=True)
#f2 = xrblc[...,inu]
#f2 = f2 - f2.mean()
##f2 = xf.sfr_lc
##f2 = f2 - f2.mean(axis=(0,1), keepdims=True)
#pkx, kpx, kzx, pkvarx, (kpb, kzb), _ = pk2d(f1, xf.Lpix, field2=f2, dkz_fct=3)
#pkx[0,0] = np.nan
##pkx[:,kzx<kmin] = np.nan
##%%
#plt.figure(dpi=150, figsize=[10, 3])
#plt.subplot(121)
#plt.imshow(np.abs(pkx).T, origin='lower', norm=LogNorm(), extent=[kpb[0], kpb[-1], kzb[0], kzb[-1]], aspect='auto');
#plt.xlabel(r'$k_\perp/{\rm Mpc^{-1}}$')
#plt.ylabel(r'$k_\parallel/{\rm Mpc^{-1}}$')
##plt.imshow(pkx.T, origin='lower');
#plt.colorbar()
#plt.title(r'$|P^{\rm HI^2\times XRB}(k_\perp, k_\parallel)|$')
#plt.subplot(122)
#plt.imshow(np.abs(pkx/np.sqrt(pkvarx)).T, origin='lower', extent=[kpb[0], kpb[-1], kzb[0], kzb[-1]], aspect='auto');
#plt.xlabel(r'$k_\perp/{\rm Mpc^{-1}}$')
#plt.ylabel(r'$k_\parallel/{\rm Mpc^{-1}}$')
#plt.colorbar()
#plt.title('SNR')
##%%
#plt.figure(figsize=[15, 3])
#plt.subplot(121)
#plot_pk(kpx, pkx[:,0], np.sqrt(pkvarx[:,0]))
#plt.xlabel(r'$k_\perp/{\rm Mpc^{-1}}$')
#plt.ylabel(r'$|P^{\rm HI\times XRB}(k_\perp, k_\parallel)|$')
#plt.title('First $k_\parallel$ bin')
#plt.subplot(122)
#plt.errorbar(kpx[1:], pkx[1:,1], np.sqrt(pkvarx[1:,1]))
#plt.xlabel(r'$k_\perp/{\rm Mpc^{-1}}$')
#plt.ylabel(r'$P^{\rm HI\times XRB}(k_\perp, k_\parallel)$')
#plt.title('Second $k_\parallel$ bin')
##%%
#plt.figure(dpi=150)
#extent = [kpb[0], kpb[-1], xf.cosmo.k2dnu(z0, kzb[0]), xf.cosmo.k2dnu(z0, kzb[-1])]
#plt.imshow(np.abs(pkx).T, origin='lower', norm=LogNorm(), extent=extent, aspect='auto')
#plt.title(r'$|P^{\rm HI\times XRB}(k_\perp, k_\parallel)|$')
#plt.xlabel(r'$k_\perp/{\rm Mpc^{-1}}$')
#plt.ylabel(r'$\Delta\nu/MHz$')
#plt.colorbar()
#plt.figure()
#plt.imshow(pkx.T>0, origin='lower');
##%%
##%%
##%%
##%%
##%%
##%%
##%%
#f1 = Tblc - Tblc.mean(axis=(0,1), keepdims=True)
#f2 = xf.sfr_lc
#f2 = f2 - f2.mean(axis=(0,1), keepdims=True)
#pkx, kx, pkx_var = pk1d(f1, 3.0, field2=f2)
##%%
#pos = pkx>0
#plt.errorbar(kx, np.abs(pkx), np.sqrt(pkx_var), c='orange', alpha=0.6)
#plt.plot(kx[pos], np.abs(pkx[pos]), 'r.', label='positive')
#plt.plot(kx[~pos], np.abs(pkx[~pos]), 'b.', label='negative')
#plt.xscale('log')
#plt.yscale('log')
##%%
#f1 = Tblc - Tblc.mean(axis=(0,1), keepdims=True)
#f2 = xf.sfr_lc.mean(axis=-1)
#f2 = f2 - f2.mean()
#pkx, kpx, kzx, pkvarx, (kpb, kzb), _ = pk2d(f1, 3.0, field2=f2)
#pkx[0,0] = np.nan
##%%
#from matplotlib.colors import LogNorm
#plt.figure()
#plt.imshow(np.abs(pkx).T, origin='lower', norm=LogNorm(), extent=[kpb[0], kpb[-1], kzb[0], kzb[-1]])
#plt.colorbar()
##%%
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
