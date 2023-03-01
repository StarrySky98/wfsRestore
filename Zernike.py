import numpy as np
import matplotlib.pyplot as plt
import aotools
from scipy import fftpack


def zernike_data(N):
    coords = (np.arange(N) - N / 2. + 0.5) / (N / 2.)
    X, Y = np.meshgrid(coords, coords)
    #转化为极坐标
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    zernike = np.ones((36, N, N))
    zernike[1] = R * np.cos(theta)
    zernike[2] = R * np.sin(theta)
    zernike[3] = np.power(R, 2)-1
    zernike[4] = np.power(R, 2) * np.cos(2*theta)
    zernike[5] = np.power(R, 2) * np.sin(2*theta)
    zernike[6] = (3 * np.power(R, 2)-2)*R * np.cos(theta)
    zernike[7] = (3 * np.power(R, 2)-2)*R * np.sin(theta)
    zernike[8] = 6 * np.power(R, 4)-6*np.power(R, 2)+1
    zernike[9] = np.power(R, 3)*np.cos(3*theta)
    zernike[10] = np.power(R, 3)*np.sin(3*theta)
    zernike[11] = (4 * np.power(R, 2)-3) * np.power(R, 2) * np.cos(2*theta)
    zernike[12] = (4 * np.power(R, 2)-3) * np.power(R, 2) * np.sin(2*theta)
    zernike[13] = (10 * np.power(R, 4) - 12*np.power(R, 2)+3) * R * np.cos(theta)
    zernike[14] = (10 * np.power(R, 4) - 12*np.power(R, 2)+3) * R * np.sin(theta)
    zernike[15] = 20*np.power(R, 6) - 30*np.power(R, 4) + 12*np.power(R, 2) - 1
    zernike[16] = np.power(R, 4)*np.cos(4*theta)
    zernike[17] = np.power(R, 4)*np.sin(4*theta)
    zernike[18] = (5 * np.power(R, 2)-4) * np.power(R, 3) * np.cos(3*theta)
    zernike[19] = (5 * np.power(R, 2)-4) * np.power(R, 3) * np.sin(3*theta)
    zernike[20] = (15 * np.power(R, 4)-20 * np.power(R, 2)+6) * np.power(R, 2) * np.cos(2*theta)
    zernike[21] = (15 * np.power(R, 4)-20 * np.power(R, 2)+6) * np.power(R, 2) * np.sin(2*theta)
    zernike[22] = (35 * np.power(R, 6)-60 * np.power(R, 4)+30*np.power(R, 2)-4) * R * np.cos(theta)
    zernike[23] = (35 * np.power(R, 6)-60 * np.power(R, 4)+30*np.power(R, 2)-4) * R * np.cos(theta)
    zernike[24] = 70*np.power(R, 8)-140*np.power(R, 6)+90*np.power(R, 4)-20*np.power(R, 2)+1
    zernike[25] = np.power(R, 5)*np.cos(5*theta)
    zernike[26] = np.power(R, 5)*np.sin(5*theta)
    zernike[27] = (6 * np.power(R, 2)-5) * np.power(R, 4) * np.cos(4*theta)
    zernike[28] = (6 * np.power(R, 2)-5) * np.power(R, 4) * np.sin(4*theta)
    zernike[29] = (21 * np.power(R, 4)-30*np.power(R, 2)+10) * np.power(R, 3) * np.cos(3*theta)
    zernike[30] = (21 * np.power(R, 4)-30*np.power(R, 2)+10) * np.power(R, 3) * np.sin(3*theta)
    zernike[31] = (56 * np.power(R, 6)-105*np.power(R, 4)+60 * np.power(R, 2)-10) * np.power(R, 2) * np.cos(2*theta)
    zernike[32] = (56 * np.power(R, 6)-105*np.power(R, 4)+60 * np.power(R, 2)-10) * np.power(R, 2) * np.sin(2*theta)
    zernike[33] = (126 * np.power(R, 8)-280*np.power(R, 6)+210*np.power(R, 4)-60*np.power(R, 2)+5) * R * np.cos(theta)
    zernike[34] = (126 * np.power(R, 8)-280*np.power(R, 6)+210*np.power(R, 4)-60*np.power(R, 2)+5) * R * np.cos(theta)
    zernike[35] = 252*np.power(R, 10)-630*np.power(R, 8)+560*np.power(R, 6)-210*np.power(R, 4)+30 * np.power(R, 2)-1
    mask = aotools.circle(N/2, N)
    for i in range(36):
        zernike[i] = np.multiply(zernike[i], mask)
    return zernike


def compute_phase(zernike, N, coff):
    mask = aotools.circle(N/2, N)
    phase = np.zeros([N, N])
    for i in range(36):
        phase = phase + np.multiply(mask, zernike[i]*coff[i])
    return phase


def compute_psf(phase):
    phase_len = len(phase)
    p = aotools.circle(phase_len/2, phase_len)
    F1 = fftpack.fft2(p * np.exp(1j* phase))
    F2 = fftpack.fftshift(F1)
    psf = np.abs(F2)**2
    return psf



if __name__ == '__main__':

    N = 128
    np.random.seed(seed=0)
                           # um
    n_zernike = 36
    o_zernike = []                                   # Zernike polynomial radial Order, see J. Noll paper :

    c_zernike = (2 * np.around(np.random.random( n_zernike), 3) - 1)/2

    print(c_zernike.shape)
    mask = aotools.circle(N / 2, N)
    coff = c_zernike
    print(coff)
    zernike = zernike_data(N)
    phase = compute_phase(zernike,N,coff)
    psf = compute_psf(phase)
    plt.figure()
    plt.imshow(psf)
    plt.colorbar()
    plt.show()
    c_zernike = 2 * np.around(np.random.random((1000, n_zernike)), 3) - 1
    for i in range(5):
        print(i)

        # print(c_zernike.shape)
        coff = c_zernike[i]
        # print(coff)
        zernike = zernike_data(N)
        phase = compute_phase(zernike, N, coff)
        psf = compute_psf(phase )
        plt.imshow(psf)
        plt.colorbar()
        plt.title("image_in")
        plt.savefig('./show/image_'+str(i)+'in.png')
        plt.show()




