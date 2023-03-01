import torch
from vggnet import VGG
import time
import matplotlib.pyplot as plt
import torch.utils.data as Data
from dataset import psf_dataset, Normalize, ToTensor
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
import os
from sklearn.metrics import mean_squared_error

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from Unet import UNet
from Unet_pp import NestedUNet

althorm = "Unetpp"
model_dir = 'model-0.5/model1-35-0.5-'
eval_dir = 'data-0.5-eval'
dataset_dir="data-0.5"


def test(althorm, model_dir, eval_dir, dataset_dir,pretreatment,size):
    if althorm=="Unet":
        net = UNet(n_channels_in=2, n_channels_out=1)
    elif althorm =="Unetpp":
        net = NestedUNet()
    else:
        pass
    batchsize = 1
    lable = []
    rmse = []
    pre = []
    net.load_state_dict(torch.load("./"+model_dir+althorm+"/model-160.pkl", map_location=torch.device('cpu')))
    test_dataset = psf_dataset(root_dir=dataset_dir, size=size, pretreatment=pretreatment,
                               transform=transforms.Compose([Normalize(), ToTensor()]))
    test_dataloader = DataLoader(test_dataset, batch_size=batchsize, num_workers=4, shuffle=False)
    for test_batch, test_batch_data in enumerate(test_dataloader):
        test_x = test_batch_data['image']
        test_y_pred = net(test_x)
        si_rmse = np.sqrt(mean_squared_error(test_y_pred.detach().numpy()[0][0], test_batch_data['phase'].numpy()[0][0]))

        print(si_rmse)
        rmse.append(si_rmse)
        pre.append(test_y_pred.detach().numpy()[0][0])
        lable.append(test_batch_data['phase'].numpy()[0][0])
    if os.path.exists(eval_dir) == False:
        os.makedirs(eval_dir)
    np.save(eval_dir+"/pre-phase-"+althorm+".npy", np.array(pre))
    np.save(eval_dir+"/rmse-phase-"+althorm+".npy", np.array(rmse))
    np.save(eval_dir+"/lable-phase.npy",np.array(lable))


if __name__ == '__main__':
    test()
    print("avg:", np.sum(np.load("./"+eval_dir+"/rmse-"+althorm+".npy"))/1000)