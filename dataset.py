import torch
from astropy.io import fits
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import math
from torchvision import transforms
import matplotlib.pyplot as plt


class psf_dataset(Dataset):

    def __init__(self, root_dir, size, pretreatment, transform=None):
        self.size = size
        self.root_dir = root_dir
        self.transform = transform
        self.pretreatment = pretreatment

    def __len__(self):
        return self.size

    def __getitem__(self, id):

        if self.root_dir =='./data-0.5/test/':
            sample_name = self.root_dir + 'psf_' + str(int(40000+id)) + '.fits'
        elif self.root_dir == 'D:/wfsExperiment/upload/':
            sample_name = self.root_dir + 'psf.fits'
        else:
            sample_name = self.root_dir + 'psf_' + str(int(id)) + '.fits'
        sample_hdu = fits.open(sample_name)
        coff = sample_hdu[0].data.astype(np.float32)
        image = np.stack((sample_hdu[2].data, sample_hdu[3].data)).astype(np.float32)

        phase = sample_hdu[1].data.astype(np.float32)

        sample = {'coff':coff,'phase': phase, 'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Normalize(object):
    def __call__(self, sample, pretreatment=0):
        coff, phase, image = sample['coff'], sample['phase'], sample['image']
        image[0] = minmax(image[0], pretreatment)
        image[1] = minmax(image[1], pretreatment)
        return {'coff': coff, 'phase': phase, 'image': image}


def minmax(array, pretreatment):
    if pretreatment == 0:
        a_min = np.min(array)
        a_max = np.max(array)
        res = (array - a_min) / (a_max - a_min)
    elif pretreatment == 1:
        a_min = np.min(array)
        a_max = np.max(array)
        res = math.sqrt((array - a_min) / (a_max - a_min))
    else:
        a_min = np.min(array)
        a_max = np.max(array)
        res = math.log((array - a_min) / (a_max - a_min))
    return res


class ToTensor(object):
    def __call__(self, sample):
        coff, phase, image = sample['coff'], sample['phase'], sample['image']
        phase = np.expand_dims(phase,axis=0)
        return {'coff':torch.from_numpy(coff),'phase': torch.from_numpy(phase), 'image': torch.from_numpy(image)}





if __name__ == '__main__':
    dataset = psf_dataset(root_dir='D:/wfsExperiment/upload/',size=1,pretreatment=0,transform=transforms.Compose([Normalize(), ToTensor()]))
    train_dataloader = DataLoader(dataset, batch_size=1, num_workers=4)
    # test_dataset = psf_dataset(root_dir='./data/test/', size=1000, transform=transforms.Compose([Normalize(), ToTensor()]))
    # test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=4)
    for i_batch,sample in enumerate(train_dataloader):
        coff = sample['coff'][0]
        print(coff)
        if i_batch==6:
            break
    # for i_batch,sample in enumerate(test_dataloader):
    #     coff = sample['coff'][0]
    #     print(coff)
    #
    #     phase = sample['phase'][0][0]
    #     image_in = sample['image'][0][0]
    #     print(sample['image'].shape)
    #     image_out = sample['image'][0][1]
    #
    #     f, axarr = plt.subplots(4, 1, figsize=(15, 10))
    #     im1 = axarr[0].imshow(phase, cmap=plt.cm.jet)
    #     # im1.set_clim(-np.pi, np.pi)
    #     axarr[0].set_title("Phase")
    #     plt.colorbar(im1, ax=axarr[0], fraction=0.046)
    #     im2 = axarr[1].imshow(image_in, cmap=plt.cm.jet)
    #     axarr[1].set_title("In")
    #     plt.colorbar(im2, ax=axarr[1], fraction=0.046)
    #     im3 = axarr[2].imshow(image_out, cmap=plt.cm.jet)
    #     axarr[2].set_title("Out")
    #     plt.colorbar(im3, ax=axarr[2], fraction=0.046)
    #     im3 = axarr[3].plot(coff)
    #     axarr[3].set_title("coff")
    #     plt.show()