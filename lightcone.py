#%%
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
import warnings
import h5py as h5
from interp_3d_cubic import interp3d_cubic
from utils import cosmo_utils, fft_upgrade, linear_interp_2p, linear_interp_idx
import time

def get_k_vec(th, phi):
    kx = np.sin(th)*np.cos(phi)
    ky = np.sin(th)*np.sin(phi)
    kz = np.cos(th)
    return kx, ky, kz

def get_los(th, phi, x0, y0, d=0.001, chunk_size=10000):
    kx, ky, kz = get_k_vec(th, phi)
    r = np.arange(chunk_size)*d
    pos0 = np.asfortranarray([kx*r+x0, ky*r+y0, kz*r]).T # N, 3
    pos1 = pos0%1
    pos_arr = []
    r_arr = []
    flag = True
    pos0_fix = np.array([x0, y0, 0.0])
    pos_fix = []
    while flag:
        i_break = np.where(np.linalg.norm(pos0-pos1, axis=-1)>5*d)[0]
        if len(i_break) == 0:
            pos_arr.append(pos0)
            r_arr.append(r)
            pos_fix.append(pos0_fix)
            break
        i_break = i_break[0]
        pos_arr.append(pos0[:i_break])
        r_arr.append(r[:i_break])
        pos_fix.append(pos0_fix)
        dpos = pos1[i_break] - pos0[i_break]
        pos0 = pos0[i_break:] + dpos
        pos0_fix = pos0_fix + dpos
        pos1 = pos0%1
        r = r[i_break:]
    return pos_arr, r_arr, pos_fix

def get_dist_perp(dpos, k_vec):
    '''
    dpos: (N1, N2, ..., 3)\n
    '''
    dpos_para = dpos@k_vec
    dpos_len = np.linalg.norm(dpos, axis=-1)
    return dpos_len - np.abs(dpos_para), dpos_len

def get_dist(pos0, pos1, r0, r1, k_vec, boxsize=1):
    '''
    pos0: (N, 3)\n
    pos1: (M, 3)\n
    '''
    dpos = pos0[:,None] - pos1 # (N, M, 3)
    dr = r0[:,None] - r1

    dpos = dpos%boxsize
    cont = dr<boxsize
    if np.any(cont):
        dpos_len = np.zeros(dpos.shape[:2], dtype=np.float64) + np.inf # N, M
        ii = 0
        for i1, i2, i3 in np.ndindex(2, 2, 2):
            this_dpos = dpos.copy()
            this_dpos[...,0] -= boxsize*i1
            this_dpos[...,1] -= boxsize*i2
            this_dpos[...,2] -= boxsize*i3
            this_dpos_perp, this_dpos_len = get_dist_perp(this_dpos, k_vec) # N, M
            mask = (this_dpos_perp<boxsize*1e-5) & cont
            this_dpos_len[mask] = np.inf
            dpos_len = np.minimum(dpos_len, this_dpos_len)
            ii += 1
    else:
        dpos[dpos>boxsize/2.0] -= boxsize
        dpos_len = np.linalg.norm(dpos, axis=-1)

    return dpos_len.min(axis=-1)

def select_line(th, phi, pos_arr, r_arr, dist_min, verbose=True):
    n = len(pos_arr)

    #kx = np.sin(th)*np.cos(phi)
    #ky = np.sin(th)*np.sin(phi)
    #kz = np.cos(th)
    #k_vec = np.array([kx, ky, kz])

    i_block = []
    i_cut = []
    n_total = 0
    k_vec = np.array(get_k_vec(th, phi))
    for ii in range(n):
        imax = pos_arr[ii].shape[0]
        this_min = np.inf
        jmin = None
        for jj in range(ii):
            dpos = get_dist(pos_arr[ii], pos_arr[jj], r_arr[ii], r_arr[jj], k_vec, boxsize=1)
            idx = np.where( dpos<dist_min )[0]
            if dpos.min() < this_min:
                this_min = dpos.min()
                jmin = jj
            if len(idx) != 0: imax = min(imax, idx[0])
        if jmin is not None:
            if verbose: print('Minimum distance for line %d: %s from line %d'%(ii, this_min, jmin))
        if verbose: print('Select %d/%d for line %d'%(imax, pos_arr[ii].shape[0], ii))
        n_total += imax
        if imax != 0:
            i_block.append(ii)
            i_cut.append(imax)
        if imax < pos_arr[ii].shape[0]:
            break
    else:
        warnings.warn('All points are selected, the lightcone can be longer')
    if verbose: print('Select %d points'%n_total)
    return i_block, i_cut

def get_dist_min(th, phi, pos_arr, r_arr, verbose=True):
    n = len(pos_arr)

    dist_min = []
    k_vec = np.array(get_k_vec(th, phi))
    for ii in range(n):
        imax = pos_arr[ii].shape[0]
        this_min = np.inf + np.zeros_like(pos_arr[ii][:,0])
        for jj in range(ii):
            dpos = get_dist(pos_arr[ii], pos_arr[jj], r_arr[ii], r_arr[jj], k_vec, boxsize=1)
            this_min = np.minimum(this_min, dpos)
        dist_min.append(this_min)
    return dist_min


def get_flat_sky_lc(th, phi, x0, y0, r_arr, dx, dy, nx, ny, parallel_los=False):
    '''
    dx, dy in deg\n
    '''
    x = np.arange(nx)*np.deg2rad(dx)
    y = np.arange(ny)*np.deg2rad(dy)
    x = x - x.mean()
    y = y - y.mean()
    xm, ym = np.meshgrid(x, y, indexing='ij')
    shp = xm.shape
    xm = xm.reshape(-1)
    ym = ym.reshape(-1)
    zm = np.sqrt(1 - xm**2 - ym**2)
    grid = np.asfortranarray([xm, ym, zm]).T # Npix, 3
    rot = Rotation.from_euler('yz', [th, phi], degrees=False)
    grid = rot.apply(grid) # Npix, 3
    kx, ky, kz = get_k_vec(th, phi)
    pos0 = np.array([r_arr[0]*kx, r_arr[0]*ky, 0.0])
    dpos = np.array([x0, y0, 0.0]) - pos0
    if parallel_los:
        vec_los = np.array([kx, ky, kz])
        grid = grid[:,None,:]*r_arr[0] + vec_los[None,:]*(r_arr-r_arr[0])[:,None]
    else:
        grid = grid[:,None,:] * r_arr[:,None] # Npix, nr, 3

    #imin = np.argmin(xm**2+ym**2)
    #print(pos0, grid[imin,0])

    grid = grid + dpos
    return grid.reshape(*shp, -1, 3)


#%%
# Check the los
if __name__ == '__main__':
    seed = 101
    np.random.seed(seed)
    r_min = 0.35

    th = np.arccos(np.random.uniform(0, 1))
    phi = np.random.rand()*2*np.pi
    x0 = np.random.rand()
    y0 = np.random.rand()

    pos_arr, r_arr, pos_fix = get_los(th, phi, x0, y0, d=0.005, chunk_size=10000)

    kx, ky, kz = get_k_vec(th, phi)
    #for ii in range(len(pos_arr)):
    #    this_pos = np.asfortranarray([r_arr[ii]*kx, r_arr[ii]*ky, r_arr[ii]*kz]).T + pos_fix[ii]
    #    assert np.abs(this_pos-pos_arr[ii]).max()<1e-10

    #plt.figure()
    #plt.subplot(121)
    #for ii in pos_arr:
    #    l = plt.plot(ii[:,0], ii[:,2])
    ##    print(l[0].get_color())
    #plt.subplot(122)
    #for ii in pos_arr:
    #    plt.plot(ii[:,1], ii[:,2])
    #plt.show()

    i_block, i_cut = select_line(th, phi, pos_arr, r_arr, r_min, verbose=True)

    r_sel = [r_arr[ii][:jj] for ii, jj in zip(i_block, i_cut)]
    pos_sel = [pos_arr[ii][:jj] for ii, jj in zip(i_block, i_cut)]
    r_sel = np.concatenate(r_sel)
    pos_sel = np.concatenate(pos_sel, axis=0)
    print(r_sel.max())
    
    dist_min = get_dist_min(th, phi, pos_arr, r_arr)
    for ii, jj in zip(i_block, i_cut):
        print(dist_min[ii].min(), dist_min[ii][:jj].min())

    plt.figure()
    ist = 0
    for ii in dist_min:
        ied = ist + len(ii)
        plt.plot(np.arange(ist, ied), ii)
        ist = ied
    plt.show()

#%%
# build grids and check numerically whether there are repeating grids
if __name__ == '__main__':
    th = np.deg2rad(76.0)
    phi = np.deg2rad(40.2)
    x0, y0 = 0.67129914, 0.22137547
    pos_arr, r_arr, pos_fix = get_los(th, phi, x0, y0, d=0.005, chunk_size=10000)
    
    kx, ky, kz = get_k_vec(th, phi)
    
    plt.figure()
    plt.subplot(121)
    for ii in pos_arr:
        l = plt.plot(ii[:,0], ii[:,2])
    #    print(l[0].get_color())
    plt.subplot(122)
    for ii in pos_arr:
        plt.plot(ii[:,1], ii[:,2])
    plt.show()
    
    i_block, i_cut = select_line(th, phi, pos_arr, r_arr, 0.35, verbose=True)
    
    r_lc = [r_arr[ii][:jj] for ii, jj in zip(i_block, i_cut)]
    pos_lc = [pos_arr[ii][:jj] for ii, jj in zip(i_block, i_cut)]
    r_lc = np.concatenate(r_lc)
    pos_lc = np.concatenate(pos_lc, axis=0)
    print(r_lc.max())
    
    r0 = 20.7
    nx = 64
    ny = 64
    dx = 0.5/nx
    dy = 0.5/nx
    print(np.deg2rad(dx)*nx*(r_lc.max()+r0))
    print(np.deg2rad(dy)*ny*(r_lc.max()+r0))
    grid = get_flat_sky_lc(th, phi, x0, y0, r_lc+r0, dx, dy, nx, ny)

    grid1 = get_flat_sky_lc(th, phi, x0, y0, r_lc+r0, dx, dy, nx, ny, parallel_los=True)
    dgrid = grid - grid1
    plt.figure()
    plt.subplot(121)
    plt.plot(grid[0,0,:,0], grid[0,0,:,1], alpha=0.6)
    plt.plot(grid[nx-1,0,:,0], grid[nx-1,0,:,1], alpha=0.6)
    plt.plot(grid[0,ny-1,:,0], grid[0,ny-1,:,1], alpha=0.6)
    plt.plot(grid[nx-1,ny-1,:,0], grid[nx-1,ny-1,:,1], alpha=0.6)
    plt.axis('equal')
    plt.subplot(122)
    plt.plot(grid[0,0,:,0], grid[0,0,:,2], alpha=0.6)
    plt.plot(grid[nx-1,0,:,0], grid[nx-1,0,:,2], alpha=0.6)
    plt.plot(grid[0,ny-1,:,0], grid[0,ny-1,:,2], alpha=0.6)
    plt.plot(grid[nx-1,ny-1,:,0], grid[nx-1,ny-1,:,2], alpha=0.6)
    plt.axis('equal')

    plt.figure()
    plt.plot(np.linalg.norm(dgrid[0,0], axis=-1), alpha=0.6)
    plt.plot(np.linalg.norm(dgrid[nx//2,0], axis=-1), alpha=0.6)
    plt.plot(np.linalg.norm(dgrid[0,ny//2], axis=-1), alpha=0.6)
    plt.plot(np.linalg.norm(dgrid[nx//2,ny//2], axis=-1), alpha=0.6)
    plt.figure()
    k_vec = np.array([kx, ky, kz])
    plt.plot(dgrid[0,0]@k_vec)
    plt.plot(dgrid[nx//2,0]@k_vec)
    plt.plot(dgrid[0,ny//2]@k_vec)
    plt.plot(dgrid[nx//2,ny//2]@k_vec)
    plt.show()

    grid = grid.reshape(-1, grid.shape[2], grid.shape[3])
    grid = grid%1
    nb = 128
    idx_cnt = np.zeros([len(i_block), nb, nb, nb], dtype=np.float64)
    ist = 0
    nr_trim = 3
    quit = ''
    for ii in range(len(i_block)):
        print('%d/%d'%(ii+1, len(i_block)))
        ied = ist + i_cut[ii]
        this_grid = grid[:,ist:ied,:]
        idx = np.floor(this_grid*nb).astype(np.int64)
        for i1, i2 in np.ndindex(this_grid.shape[:2]):
            if i2 < nr_trim: continue
            idx_cnt[ii,idx[i1,i2,0],idx[i1,i2,1],idx[i1,i2,2]] = 1
        print(ist, ied)
        if quit.strip()!='y':
            plt.figure()
            plt.subplot(121)
            plt.imshow(idx_cnt[:ii+1].sum(axis=(0,-1)).T, origin='lower', interpolation='none')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.subplot(122)
            plt.imshow(idx_cnt[:ii+1].sum(axis=(0,-2)).T, origin='lower', interpolation='none')
            plt.xlabel('x')
            plt.ylabel('z')
            plt.tight_layout()
            plt.show()
            quit = input()
        ist = ied
    
    print(idx_cnt.sum(axis=0).max())
    print(np.unique(idx_cnt.sum(axis=0), return_counts=True))
#%%
def damp_fft_mode(field, los, dx, para_func, perp_func):
    '''
    The para_func(k_parallel) and perp_func(k_perp) give the damp kernel for k_\parallel and k_\perp and if None, do not apply\n
    NOTE: for correct normalization, para_func(0) and perp_func(0) should be 1\n
    '''
    field = np.asarray(field)
    los = np.asarray(los)
    los = los/np.linalg.norm(los)
    assert np.ndim(field) == los.shape[0]
    if np.ndim(dx) == 0:
        dx = [dx]*los.shape[0]
    else:
        assert len(dx) == los.shape[0]
    field_fft = np.fft.fftn(field)
    k = [np.fft.fftfreq(ii, d=d)*2*np.pi for ii, d in zip(field.shape, dx)]
    kall = np.asfortranarray(np.meshgrid(*k, indexing='ij'))
    k2 = np.sum(kall**2, axis=0)
    k_para = (kall.T@los).T
    k_perp = np.sqrt(np.maximum(k2-k_para**2, 0.0))
    kernel = 1.0
    if para_func is not None:
        kernel = kernel * para_func(k_para)
    if perp_func is not None:
        kernel = kernel * perp_func(k_perp)
    field_fft = field_fft*kernel
    field_new = np.fft.ifftn(field_fft)
    if field.dtype.kind != 'c': field_new = field_new.real
    return field_new

#%%
if __name__ == '__main__':
    nx = 64
    ny = 32
    f = np.random.randn(nx, ny)
    dx = 0.2
    dy = 0.1
    dkx = 2*np.pi/(nx*dx)
    dky = 2*np.pi/(ny*dy)
    dk = np.sqrt(dkx**2+dky**2)
    los = [1.0, 0.5]
    extent = [-dx/2.0, dx*nx+dx/2.0, -dy/2.0, dy*ny+dy/2.0]
    para_func = lambda k: np.exp(-k**2/(2*dk)**2/2.0)
    perp_func = lambda k: np.exp(-k**2/(dk)**2/2.0)
    f1 = damp_fft_mode(f, los, dx, para_func, perp_func)
    f2 = damp_fft_mode(f, los, [dx, dy], para_func, perp_func)
    f3 = damp_fft_mode(f, los, [dx, dy], para_func, None)
    f4 = damp_fft_mode(f, los, [dx, dy], None, perp_func)
    plt.figure()
    plt.imshow(f.T, origin='lower', extent=extent)
    plt.figure()
    plt.imshow(f1.T, origin='lower', extent=extent[:2] + [-dx/2.0, dx*ny+dx/2.0])
    plt.plot([0, nx*dx], [0, nx*dx*los[1]/los[0]], 'r')
    plt.figure()
    plt.imshow(f2.T, origin='lower', extent=extent)
    plt.plot([0, nx*dx], [0, nx*dx*los[1]/los[0]], 'r')
    plt.figure()
    plt.imshow(f3.T, origin='lower', extent=extent)
    plt.plot([0, nx*dx], [0, nx*dx*los[1]/los[0]], 'r')
    plt.figure()
    plt.imshow(f4.T, origin='lower', extent=extent)
    plt.plot([0, nx*dx], [0, nx*dx*los[1]/los[0]], 'r')
    plt.show()
#%%
def get_RSD(delta, los, dx=None):
    delta = np.asarray(delta)
    los = np.asarray(los)
    los = los/np.linalg.norm(los)
    assert np.ndim(delta) == los.shape[0]
    if dx is None: dx = 1.0
    if np.ndim(dx) == 0:
        dx = [dx]*los.shape[0]
    else:
        assert len(dx) == los.shape[0]
    delta_fft = np.fft.fftn(delta)
    k = [np.fft.fftfreq(ii, d=d)*2*np.pi for ii, d in zip(delta.shape, dx)]
    kall = np.asfortranarray(np.meshgrid(*k, indexing='ij'))
    k2 = np.sum(kall**2, axis=0)
    #print(k2[0,0,0])
    k2[0,0,0] = np.nan
    k_para = (kall.T@los).T
    mu2 = k_para**2/k2
    mu2[0,0,0] = 0.0
    delta_vel = np.fft.ifftn(delta_fft*mu2)
    #print(np.abs(delta_vel.imag).max())
    return delta_vel.real
#%%
if __name__ == '__main__':
    from scipy.io import loadmat
    ic_num = 1040
    z = 10
    d = loadmat('/rds/user/fd426/hpc-work/21cm/intermediate/xray_analysis/sim_fd_xray_IC_1040/Tb_fields/Tb_field_%d_fd_xray_IC_%d.mat'%(z, ic_num))
    Tlin = d['Tlin']
    Tb = d['Tb']
    delta_DM = loadmat('/rds/project/rds-PJtLerV8oy0/JVD_21cmSPACE_Precomputed_Grids/IC/128/delta%d.mat'%ic_num)['delta']
    Dz = loadmat('/home/fd426/21cmSPACE/Dz.mat')['D'][0]
    z4D = loadmat('/home/fd426/21cmSPACE/zs_for_D.mat')['zs'][0]
    delta_DM = delta_DM/np.interp(40, z4D, Dz) * np.interp(z, z4D, Dz)
    delta_DM = np.clip(delta_DM, -0.999, 1.3)
    delta_v = get_RSD(delta_DM, [0.0, 0.0, 1.0])
    Tlin_test = Tb*(delta_v+1)
    print(np.abs(Tlin_test-Tlin).max())
#%%
#%%

class lightcone_utils(cosmo_utils):
    def __init__(self, z_fields, fields, boxsize, params_file=None, **itp_table_kwargs):
        '''
        NOTE: at present only support cubic box and therefore boxsize must be scalar\n
        NOTE: z in the class represent redshift and the los length is r\n
        '''
        cosmo_utils.__init__(self, params_file=params_file, **itp_table_kwargs)
        self.z_fields = np.asarray(z_fields)
        fields = np.asarray(fields)
        assert self.z_fields.shape[0] == fields.shape[0]
        self.nz = self.z_fields.shape[0]
        if not np.all(np.diff(self.z_fields)>0):
            isort = np.argsort(self.z_fields)
            self.z_fields = self.z_fields[isort]
            fields = fields[isort]
        self.boxsize = boxsize
        self.fields = fields
        self.dx = np.array([boxsize/np.float64(ii) for ii in self.fields.shape[1:]])
        self.grid_box = None

    def set_los(self, th, phi, x0, y0, z0, drc, dx, dy, nx, ny, parallel_los=False):
        '''
        th, phi in rad and dx, dy in deg\n
        '''
        assert np.log(z0+1) < self.lnzp1.max()
        assert np.all(np.diff(drc)>0)
        r0 = self.get_rc(z0)
        self.z_lc = self.drc2z(z0, drc)
        print('Redshift: %s to %s'%(self.z_lc.min(), self.z_lc.max()))
        if self.z_lc.min()<self.z_fields.min(): warnings.warn('Minimum redshift for lightcone %s is lower than lowest redshift for fields %s'%(self.z_lc.min(), self.z_fields.min()))
        if self.z_lc.max()>self.z_fields.max(): warnings.warn('Maximum redshift for lightcone %s is higher than highest redshift for fields %s'%(self.z_lc.max(), self.z_fields.max()))
        idx0_z, idx1_z = linear_interp_idx(self.z_lc, self.z_fields, side='right')
        idx_sec = np.where(np.diff(idx0_z)>0)[0]+1
        idx_sec = np.append(0, idx_sec)
        idx_sec = np.append(idx_sec, len(idx0_z))
        self.idx_sec = idx_sec
        self.idxh_z = idx0_z[self.idx_sec[:-1]]
        self.idxl_z = idx1_z[self.idx_sec[:-1]]

        #print('============= Only for check =============')
        #assert np.array_equal(idx0_z[idx_sec[:-1]], np.unique(idx0_z))
        #assert np.array_equal(idx1_z[idx_sec[:-1]], np.unique(idx1_z))
        #for ist, ied in zip(idx_sec[:-1], idx_sec[1:]):
        #    assert np.all(idx0_z[ist:ied]==idx0_z[ist])
        #    assert np.all(idx1_z[ist:ied]==idx1_z[ist])
        #print('============= Only for check =============')

        self.rc_lc = r0 + drc
        trans_dist = np.deg2rad(np.sqrt((nx*dx)**2+(ny*dy)**2))
        trans_dist = self.rc_lc.max()*trans_dist
        print('Maximum transverse distance: %s Mpc (%.1f%% of the boxsize)'%(trans_dist, trans_dist/self.boxsize*100))
        trans_dist = np.deg2rad(max(dx, dy))
        trans_dist = self.rc_lc.max()*trans_dist
        print('Maximum pixel size: %s Mpc'%trans_dist)
        self.grid_lc = get_flat_sky_lc(th, phi, x0, y0, self.rc_lc, dx, dy, nx, ny, parallel_los=parallel_los) # Nx, Ny, nr, 3 
        #_, _, coord_z0 = get_k_vec(th, phi)
        self.los = np.array(get_k_vec(th, phi))
        coord_z0 = self.los[-1]*self.rc_lc[0]
        self.grid_box = self.grid_lc.copy()
        self.grid_box[...,-1] -= coord_z0
        self.grid_box = self.grid_box%self.boxsize
        self.x_lc = np.arange(self.grid_box.shape[0])*dx
        self.x_lc = self.x_lc - self.x_lc.mean()
        self.y_lc = np.arange(self.grid_box.shape[1])*dy
        self.y_lc = self.y_lc - self.y_lc.mean()
        self.dx_lc = dx
        self.dy_lc = dy
        self.extent_lc = [self.x_lc[0]-dx/2.0, self.x_lc[-1]+dx/2.0, self.y_lc[0]-dy/2.0, self.y_lc[-1]+dy/2.0]

    def interp_lc(self, upgrade=None, kernel_para=None, kernel_perp=None):
        assert self.grid_box is not None, 'Run set_los before interpolate'
        fields_lc = np.zeros(self.grid_lc.shape[:-1], dtype=self.fields.dtype)
        t0 = time.time()
        il = self.idxl_z[0]
        fl = self.fields[il]
        if upgrade is not None:
            fl = fft_upgrade(fl, upgrade)[0]
            d = self.dx/upgrade
        else:
            d = self.dx
        if kernel_para is None or kernel_perp is None:
            fl = damp_fft_mode(fl, self.los, d, kernel_para, kernel_perp)
        #fl = extend_field(fl)
        #coords = [np.arange(ii)/(ii-1.0)*self.boxsize for ii in fl.shape]
        refresh = False
        for ii in range(self.idx_sec.shape[0]-1):
            ist = self.idx_sec[ii]
            ied = self.idx_sec[ii+1]
            print('%d/%d redshift block (redshift %.3f to %.3f): %.2f second'%(ii+1, self.idx_sec.shape[0]-1, self.z_lc[ist], self.z_lc[ied-1], time.time()-t0))
            ih, il = self.idxh_z[ii], self.idxl_z[ii]
            print('Use snap %d and %d'%(il, ih))
            #if refresh: print('=============')
            fh = self.fields[ih]
            if refresh: fl = self.fields[il]
            if upgrade is not None:
                fh = fft_upgrade(fh, upgrade)[0]
                if refresh: fl = fft_upgrade(fl, upgrade)[0]
            if kernel_para is None or kernel_perp is None:
                fh = damp_fft_mode(fh, self.los, d, kernel_para, kernel_perp)
                if refresh: fl = damp_fft_mode(fl, self.los, d, kernel_para, kernel_perp)
            #fh = extend_field(fh)
            #if refresh: fl = extend_field(fl)
            this_grid = self.grid_box[:,:,ist:ied,:]
            
            #fieldh = RegularGridInterpolator(coords, fh, bounds_error=True)(this_grid)
            #fieldl = RegularGridInterpolator(coords, fl, bounds_error=True)(this_grid)
            fieldh = interp3d_cubic(fh, this_grid, d)
            fieldl = interp3d_cubic(fl, this_grid, d)
            fields_lc[...,ist:ied] = linear_interp_2p(self.z_lc[ist:ied], self.z_fields[ih], fieldh, self.z_fields[il], fieldl)
            if ii != self.idx_sec.shape[0]-2 and self.idxl_z[ii+1] != self.idxh_z[ii]:
                refresh = True
            else:
                refresh = False
                fl = fh
        self.fields_lc = fields_lc
#%%
if __name__ == '__main__':
    from powerbox import PowerBox, LogNormalPowerBox
    import camb
    from camb import model, initialpower
    from scipy.interpolate import UnivariateSpline
    from get_power import pk1d
    import h5py as h5

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.InitPower.set_params(ns=0.965)
    #Note non-linear corrections couples to smaller scales than you want
    z_camb = np.logspace(0, 2, 101) - 1
    z_camb[0] = 0.0
    pars.set_matter_power(redshifts=z_camb, kmax=10.0)
    
    #Linear spectra
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    k_in, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=10.0, npoints = 200)
    sigma8 = results.get_sigma8() # in order of increasing time (decreasing redshift)
    # reorder
    D = sigma8[::-1]/sigma8[-1]
    Dfunc = lambda z: np.exp( np.interp(np.log(z+1), np.log(z_camb+1), np.log(D)) )
    k_in = k_in * pars.h
    pk0 = pk[0] / pars.h**3
    #pk0[k_in<0.3] = 1e-10*pk0.max()

    z_piv = 5.0
    D_piv = Dfunc(z_piv)

    N = 128
    dx = 3.0
    seed = 1010
    dk = 2*np.pi/(N*dx)

    k_piv = N*dk/2.0 # smooth out the grid
    print('k_piv: %s'%k_piv)
    pk0 = pk0 * np.exp(-k_in**2/2.0/k_piv**2)

    pk_itp_loglog = UnivariateSpline(np.log10(k_in), np.log10(pk0*D_piv**2), s=0, k=3, ext=1)

    lnpb = LogNormalPowerBox(N=N, dim=3, pk = lambda k: 10**pk_itp_loglog(np.log10(k)), boxlength=dx*N, seed=seed)
    delta = lnpb.delta_x()



    z_snap = np.linspace(z_piv, 15, 11)
    D_snap = Dfunc(z_snap)
    fields = np.array([delta*ii/D_piv for ii in D_snap])


    kbins = np.geomspace(dk/2.0, N*dk/2.0, 15)
    pk, k, pkvar = pk1d(delta, dx=dx, kbins=kbins, dk_fct=1.0)
    plt.figure()
    plt.errorbar(k, pk/D_piv**2, np.sqrt(pkvar)/D_piv**2, ls='--')
    plt.plot(k, 10**pk_itp_loglog(np.log10(k))/D_piv**2)
    for ii in range(0, z_snap.shape[0], 4):
        pk, k, _ = pk1d(fields[ii], dx=dx, kbins=kbins, dk_fct=1.0)
        plt.plot(k, pk/D_snap[ii]**2, '-', alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    print('Initialize light-cone')
    lu = lightcone_utils(z_snap, fields, lnpb.boxlength)

    print('Set line of sight')
    drc = np.linspace(0, 1500, 1501)
    z0 = 6.0
    idx = -1
    dx = 0.005
    dy = 0.005
    nx = 64
    ny = 64
    with h5.File('/home/fd426/xray_ana/lc_params/r_min_0.45_seed_101.hdf5', 'r') as filein:
        lu.set_los(filein['th'][idx], filein['phi'][idx], filein['x0'][idx]*lnpb.boxlength, filein['y0'][idx]*lnpb.boxlength, z0, drc, dx, dy, nx, ny, parallel_los=True)
        print(filein['r_max'][idx])

    print('Interpolate')
    lu.interp_lc(upgrade=2)
    lc = lu.fields_lc.copy()



    # check the 2D power spectrum with limber approximation
    z_min = 6.0
    z_max = 12
    valid = (lu.z_lc>=z_min) & (lu.z_lc<=z_max)
    print('Maximum radial distance: ', lu.rc_lc[valid].max()-lu.rc_lc[valid].min())

    k_cut = 1.5*2*np.pi/np.deg2rad(max(dx*nx, dy*ny))/lu.rc_lc[valid].max()
    print('k_cut: %s'%k_cut)
    print('k_min for the box: %s'%dk)
    kernel_perp = lambda k: np.array(k>k_cut, dtype=np.float64)
    lu.interp_lc(upgrade=2, kernel_perp=kernel_perp)
    lc1 = lu.fields_lc.copy()

    field = np.trapz(lc[...,valid], lu.rc_lc[valid], axis=-1)
    field1 = np.trapz(lc1[...,valid], lu.rc_lc[valid], axis=-1)

    plt.figure()
    plt.imshow(field.T, origin='lower')
    plt.figure()
    plt.imshow(field1.T, origin='lower')
    plt.show()

    pk_field, k_field, _ = pk1d(field-field.mean(), np.deg2rad([dx, dy]), get_var=False, dk_fct=1.0)
    pk_field1, k_field1, _ = pk1d(field1-field1.mean(), np.deg2rad([dx, dy]), get_var=False, dk_fct=1.0)

    D_int = np.interp(np.log(lu.z_lc[valid]+1), np.log(z_snap+1), np.log(D_snap))
    D_int = np.exp(D_int)

    integrand = D_int**2/lu.rc_lc[valid]**2
    pk_theory = np.trapz(integrand, lu.rc_lc[valid]) * pk0
    rc_lc = lu.rc_lc[valid]
    valid = (k_in*rc_lc.mean()>=k_field[0]) & (k_in*rc_lc.mean()<=k_field[-1])
    plt.figure()
    plt.loglog(k_field, pk_field)
    ylim = (pk_field[1:].min()*0.1, max(pk_field.max(), pk_theory[valid].max())*5.0)
    plt.loglog(k_field1, pk_field1, '.')
    plt.loglog(k_in[valid]*rc_lc.mean(), pk_theory[valid])
    plt.loglog(k_in[valid]*rc_lc.max(), pk_theory[valid], '--')
    plt.loglog(k_in[valid]*rc_lc.min(), pk_theory[valid], '--')
    plt.ylim(ylim)
    plt.vlines([k_piv*rc_lc.mean(), k_piv*rc_lc.min(), k_piv*rc_lc.max()], *ylim, color='b', ls='dashed')
    plt.vlines(dk*rc_lc.mean(), *ylim, color='k', ls='dashed')
    plt.vlines(k_cut*rc_lc.mean(), *ylim, color='g', ls='dashed')
    plt.vlines(2*np.pi/np.deg2rad(np.sqrt(nx*ny*dx*dy)), *ylim, color='r', ls='dashed')
    plt.show()

#%%






# %%
