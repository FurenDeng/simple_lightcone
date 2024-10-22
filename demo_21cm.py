import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import h5py as h5
from scipy.io import loadmat
from lightcone import lightcone_utils, get_RSD, get_k_vec
from glob import glob
import os
import get_power as getpk

# in this demo I generate light-cone for 21 cm brightness temperature
# NOTE: you would need to save the Tb fields WITHOUT RSD in 21cmSPACE/analysis/RunDoAnalysis_params.m

code_path = '/home/fd426/21cmSPACE'

# the line-of-sight is not along z-axis to avoid repeating structure and I cannot use the Tb cube with RSD
# load initial dark matter overdensity and growth factor for RSD calculation
IC_num = 1040
delta_DM = loadmat('/rds/project/rds-PJtLerV8oy0/JVD_21cmSPACE_Precomputed_Grids/IC/128/delta%d.mat'%IC_num)['delta']
Dz = loadmat('%s/Dz.mat'%code_path)['D'][0]
z4D = loadmat('%s/zs_for_D.mat'%code_path)['zs'][0]
# normalize by z=40
D40 = np.interp(40, z4D, Dz)
Dfunc = lambda z: np.interp(z, z4D, Dz)/D40

# load line-of-sight with minimum repeating
# NOTE: the is normalized to 1 when generating this file
idx_los = -1 # the last one is the los with least repeating
# the file name means that when the diameter of the light-cone is 0.45*boxlength, there is no repeating
# however, mild repeating seems not induce huge problem
with h5.File('./lc_params/r_min_0.45_seed_101.hdf5', 'r') as filein:
    th_los = filein['th'][idx_los]
    phi_los = filein['phi'][idx_los]
    x0_los = filein['x0'][idx_los]
    y0_los = filein['y0'][idx_los]
    r_max_los = filein['r_max'][idx_los]

vec_los = np.array(get_k_vec(th_los, phi_los))


# here is the pre-saved Tb field, filename format is 'Tb_field_%d_fd_xray_IC_%d.mat'%(redshift, IC_num)
# NOTE: Tb does not contain the RSD effect
filelist = glob('/rds/user/fd426/hpc-work/21cm/intermediate/xray_analysis/sim_fd_xray_IC_%d/Tb_fields/*'%IC_num)
zmax = 15
zmin = 7
z_Tb = [] # first load z to sort
for filename in filelist:
    z_Tb.append(float(os.path.basename(filename).split('_')[2]))
z_Tb = np.array(z_Tb)
isort = np.argsort(z_Tb)
z_Tb = z_Tb[isort]
valid = (z_Tb>=zmin) & (z_Tb<=zmax)
z_Tb = z_Tb[valid]
filelist = [filelist[ii] for ii in isort[valid]]
Tb_fields = []
pk_Tb = []
# load Tb field and add RSD effect
for zi, filename in zip(z_Tb, filelist):
    print('Load %s at redshift %d'%(filename, zi))
    this_Tb = loadmat(filename)['Tb']
    this_Tb_rsd = loadmat(filename)['Tlin'] # this field have rsd effect and use it to exame the power spectrum
    this_pk, k_Tb, _ = getpk.pk1d(this_Tb_rsd-this_Tb_rsd.mean(), 3.0, get_var=False, dk_fct=2.0)
    pk_Tb.append(this_pk)
    this_dv = get_RSD(delta_DM*Dfunc(zi), vec_los)
    this_Tb = this_Tb*(1.0+this_dv)
    Tb_fields.append(this_Tb)
Tb_fields = np.array(Tb_fields)
pk_Tb = np.array(pk_Tb)

# initialize object
boxlength = 3.0*Tb_fields.shape[-1]
params_file = '%s/Planck_parameters.mat'%code_path # cosmological parameters
lu = lightcone_utils(z_Tb, Tb_fields, boxlength, params_file=params_file)


z0 = 8 # the start redshift for the light-cone
# z=8 to z=14 is roughtly 1201 Mpc
drc = np.arange(0, 1201, 1) # the radial distance step for the light-cone, in Mpc
dx_fov = 0.01 # the angular resolution for the light-cone for x-axis, in deg
nx = 128 # the number of pixel for x-axis
dy_fov = dx_fov # for y-axis
ny = nx # for y-axis

upgrade = 4 # upgrade by fft to improve the precision of interpolation while preserve Fourier modes

print('Maximum los length: %s Mpc'%(r_max_los*lu.boxsize))
print('Current light-cone length: %.1f Mpc'%(drc[-1]-drc[0]))
lu.set_los(th_los, phi_los, x0_los*lu.boxsize, y0_los*lu.boxsize, z0, drc, dx_fov, dy_fov, nx, ny, parallel_los=False)
lu.interp_lc(upgrade=upgrade)

lc = lu.fields_lc
z_lc = lu.z_lc
rc_lc = lu.rc_lc
extent_lc = lu.extent_lc
print(lc.shape, z_lc.shape)
print(extent_lc) # in deg

# check los's
plt.figure()
plt.plot(z_lc[:100], lc[0,0,:100])
plt.plot(z_lc[:100], lc[0,60,:100])
plt.plot(z_lc[:100], lc[60,60,:100])
plt.xlabel('z')
plt.ylabel('Tb/mK')

# check 2d fields at different redshifts
plt.figure(figsize=[10, 3])
for ii, idx in enumerate([0, 100, 300]):
    plt.subplot(131+ii)
    plt.imshow(lc[:,:,idx].T, extent=extent_lc)
    plt.xlabel('x/deg')
    plt.ylabel('y/deg')
    plt.title('z=%.3f'%z_lc[idx])
plt.tight_layout()
plt.show()

# check power spectrum for individual redshift sections
for z_test in [9.0, 10.0, 11.0, 12.0]:
#for z_test in [9.0, 10.0]:
    idx_box = np.argmin(np.abs(z_Tb-z_test))
    pk_box = pk_Tb[idx_box]
    
    idx_lc = np.argmin(np.abs(z_lc-z_test))
    
    grid_size_lc = [np.deg2rad(dx_fov)*rc_lc[idx_lc], np.deg2rad(dy_fov)*rc_lc[idx_lc], drc[1]-drc[0]]
    num_sel = np.floor(lu.boxsize*0.9/grid_size_lc[-1]) # take a radial section a little bit smaller the size of input box
    num_sel = int(num_sel)
    lc_sel = lc[:,:,idx_lc-num_sel//2:idx_lc+num_sel//2]
    pk_lc, k_lc, _ = getpk.pk1d(lc_sel-lc_sel.mean(axis=(0,1), keepdims=True), grid_size_lc, get_var=False)
    
    plt.figure()
    plt.loglog(k_lc, pk_lc)
    plt.loglog(k_Tb, pk_box)
    plt.xlabel('k/Mpc^-1')
    plt.ylabel('P(k)/(mK^2 Mpc^3)')
    plt.show()

# check 2d power spectrum versus limber approximation
field_2d = np.trapz(lc, rc_lc, axis=-1)
cl_lc, l_lc, _ = getpk.pk1d(field_2d-field_2d.mean(), np.deg2rad([dx_fov, dy_fov]))

pk_itp = np.zeros(z_lc.shape+pk_Tb.shape[1:], dtype=np.float64) # nz, nk
for ik in range(k_Tb.shape[0]):
    pk_itp[:,ik] = np.exp(np.interp(z_lc, z_Tb, np.log(pk_Tb[:,ik])))
cl_box = np.trapz(pk_itp/rc_lc[:,None]**2, rc_lc, axis=0)
l_box = k_Tb*rc_lc.mean()
plt.figure()
plt.loglog(l_box, cl_box)
plt.loglog(l_lc, cl_lc)
plt.xlabel('l')
plt.ylabel('C_l/mK^2')
plt.show()



