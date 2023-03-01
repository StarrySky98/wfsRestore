import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from astropy.io import fits


def phase_graph_3D(phase, savedir):
    x = np.linspace(0, 128, 128)
    y = np.linspace(0, 128, 128)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = Axes3D(fig)
    plt.xlabel('x')
    plt.ylabel('y')
    ax.plot_surface(X, Y, phase, rstride=1, cstride=1, cmap='rainbow')
    # ax.set_zlim(-1, 1)
    plt.savefig(savedir)
    # plt.show()


def phase_graph_2D(phase, savedir):
    plt.figure()
    plt.imshow(phase)
    plt.colorbar()
    plt.savefig(savedir)
    plt.close()


if __name__ == '__main__':
    for i in range(1):
        sample_name = './data/train/psf_' + str(i) + '.fits'
        sample_hdu = fits.open(sample_name)
        coff = sample_hdu[0].data.astype(np.float32)
        image = np.stack((sample_hdu[2].data, sample_hdu[3].data)).astype(np.float32)
        phase = sample_hdu[1].data.astype(np.float32)
        image_in = image[0]
        image_out = image[1]
        phase_graph_3D(phase, './show/phase' + str(i) + '.png')
        phase_graph_3D(image_out, './show/image_out' + str(i) + '.png')
        phase_graph_3D(image_in, './show/image_in' + str(i) + '.png')
