#%%
from lightcone import *
import sys

# Monte carlo method to choose direction with most efficient volume usage
seed = 201
np.random.seed(seed)
n_test = 100000
r_min = float(sys.argv[1])
params = []
r_max = []
#for itest in tqdm(range(n_test)):
for itest in range(n_test):
    th = np.arccos(np.random.uniform(0, 1))
    phi = np.random.rand()*np.pi + np.pi/4
    x0 = np.random.rand()/2.0
    y0 = np.random.rand()/2.0
    params.append([th, phi, x0, y0])

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

    i_block, i_cut = select_line(th, phi, pos_arr, r_arr, r_min, verbose=False)

    r_sel = [r_arr[ii][:jj] for ii, jj in zip(i_block, i_cut)]
    pos_sel = [pos_arr[ii][:jj] for ii, jj in zip(i_block, i_cut)]
    r_sel = np.concatenate(r_sel)
    pos_sel = np.concatenate(pos_sel, axis=0)
    #print(r_sel.max())
    r_max.append(r_sel.max())

r_max = np.array(r_max)
params = np.array(params)
isort = np.argsort(r_max)
r_max = r_max[isort]
params = params[isort]
for ii, jj in zip(r_max[-50:], params[-50:]):
    print(ii, np.rad2deg(jj[:2]), jj[2:])
with h5.File('./lc_params/r_min_%.2f_seed_%s.hdf5'%(r_min, seed), 'w') as filein:
    filein['th'] = params[:,0]
    filein['phi'] = params[:,1]
    filein['x0'] = params[:,2]
    filein['y0'] = params[:,3]
    filein['r_max'] = r_max
    filein.attrs['r_min'] = r_min
    filein.attrs['seed'] = seed
#%%
#%%
#%%
#%%
#%%
