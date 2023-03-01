from Zernike import zernike_data, compute_phase, compute_psf
import numpy as np
import os
from astropy.io import fits


class GenPSF:
    """
    1.生成像差系数,确定像差范围,离焦量,
    2.生成相位
    3.生成PSF
    """
    def __init__(self, coff_range, n_zernike=35, train_psfs=2, test_psfs=1, N=128,  defous=0.2):
        self.n_zernike = n_zernike
        self.train_psfs = train_psfs
        self.test_psfs = test_psfs
        self.N = N
        self.coff_range = coff_range
        self.defous = defous
        self.coff = self._aberration_coff(n_zernike, train_psfs, test_psfs, coff_range)

    def _aberration_coff(self, n_zernike, train_psfs, test_psfs,coff_range):
        np.random.seed(seed=0)
        # shape:[psfs,35] 取值:[-1,1]
        c_zernike = 2 * np.around(np.random.random((train_psfs + test_psfs, n_zernike)), 3) - 1
        for j in range(train_psfs + test_psfs):
            for i in range(n_zernike):
                c_zernike[j, i] = np.around(c_zernike[j, i] * coff_range[i], 3)  # 确定系数范围

        coff = np.array([c_zernike[k, :] for k in range(train_psfs + test_psfs)])
        return coff

    def gen_save_psf(self, data_dir):
        zernike_basis = zernike_data(self.N)  # shape(36,N,N)
        defous = self.defous * zernike_basis[3]
        psfs_in = np.zeros((self.train_psfs+self.test_psfs, self.N, self.N))
        psfs_out = np.zeros((self.train_psfs+self.test_psfs, self.N, self.N))

        for i in range(self.train_psfs+self.test_psfs):
            aberrations_in = np.squeeze(np.sum(self.coff[i, :, None, None] * zernike_basis[1:, :, :], axis=0))
            # print(np.sum(self.coff[i, :, None, None] * zernike_basis[1:, :, :], axis=0).shape)
            psfs_in[i] = compute_psf(aberrations_in)
            aberrations_out = np.squeeze(aberrations_in) + defous
            psfs_out[i] = compute_psf(aberrations_out)
            if i < self.train_psfs:
                if os.path.exists(data_dir+"/train/") == False:
                    os.makedirs(data_dir+"/train/")
                outfile = data_dir+"/train/psf_" + str(i) + ".fits"
            else:
                if os.path.exists(data_dir+"/test/") == False:
                    os.makedirs(data_dir+"/test/")
                outfile = data_dir+"/test/psf_" + str(i) + ".fits"
            hdu_primary = fits.PrimaryHDU(self.coff[i, :].astype(np.float32))
            hdu_phase = fits.ImageHDU(aberrations_in.astype(np.float32), name='PHASE')
            hdu_In = fits.ImageHDU(psfs_in[i, :, :].astype(np.float32), name='INFOCUS')
            hdu_Out = fits.ImageHDU(psfs_out[i, :, :].astype(np.float32), name='OUTFOCUS')
            hdu = fits.HDUList([hdu_primary, hdu_phase, hdu_In, hdu_Out])
            hdu.writeto(outfile, overwrite=True)
            print(i)
if __name__ == '__main__':
    g=GenPSF()



# np.random.seed(seed=0)
#
# n_zernike = 35  # 第0项无意义，取了1-35项
# n_psfs = 10000
# N = 128  # 生成的尺寸
# c_zernike = 2 * np.around(np.random.random((n_psfs, n_zernike)), 3) - 1 #shape:[41000,35] 取值:[-1,1]
# # o_zernike = [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
# o_zernike = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
# for j in range(n_psfs):
#     for i in range(n_zernike):
#         c_zernike[j, i] = np.around(c_zernike[j, i] / o_zernike[i], 3)  # 确定系数范围
#
# coff = np.array([c_zernike[k, :]  for k in range(n_psfs)])
# zernike_basis = zernike_data(N)  # shape(36,N,N)
# defous = (0.25) * zernike_basis[3]
# psfs_in = np.zeros((n_psfs, N, N))
# psfs_out = np.zeros((n_psfs, N, N))
#
# for i in range(n_psfs):
#     aberrations_in = np.squeeze(np.sum(coff[i, :, None, None] * zernike_basis[1:, :, :], axis=0))
#     print(np.sum(coff[i, :, None, None] * zernike_basis[1:, :, :], axis=0).shape)
#     psfs_in[i] = compute_psf(aberrations_in)
#     aberrations_out = np.squeeze(aberrations_in) + defous
#     psfs_out[i] = compute_psf(aberrations_out)
#     if i < 9000:
#         outfile = "./data-0.5/train/psf_" + str(i) + ".fits"
#     else:
#         outfile = "./data-0.5/test/psf_" + str(i) + ".fits"
#     hdu_primary = fits.PrimaryHDU(c_zernike[i, :].astype(np.float32))
#     hdu_phase = fits.ImageHDU(aberrations_in.astype(np.float32), name='PHASE')
#     hdu_In = fits.ImageHDU(psfs_in[i, :, :].astype(np.float32), name='INFOCUS')
#     hdu_Out = fits.ImageHDU(psfs_out[i, :, :].astype(np.float32), name='OUTFOCUS')
#     hdu = fits.HDUList([hdu_primary, hdu_phase, hdu_In, hdu_Out])
#     hdu.writeto(outfile, overwrite=True)
#     print(i)
