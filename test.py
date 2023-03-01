import torch
from vggnet import VGG
from dataset import psf_dataset, Normalize, ToTensor
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from Zernike import zernike_data, compute_phase, compute_psf
import os
from sklearn.metrics import mean_squared_error
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from InceptionV3 import Net
from resnet50 import ResNet50


althorm = "our"
loss = "-smoothL1"
model_dir = './model1-35-'
eval_dir = 'data-eval'
dataset_dir="data"


def model_test(althorm, model_dir, eval_dir, dataset_dir,pretreatment,size,N):
    zernike_basis = zernike_data(N)  # shape(36,N,N)
    if althorm == "our":
        net = VGG("VGG16")
    elif althorm == "ResNet50":
        net = ResNet50()
    elif althorm == "InceptionV3":
        net = Net()
        net.eval()
    else:
        pass
    batchsize = 1
    rmse = []
    c_rmse = []
    pre = []
    lable = []
    phase_pre = []
    phase_lable = []
    net.load_state_dict(torch.load(model_dir+althorm +"/model-160.pkl", map_location=torch.device('cpu')))
    print(dataset_dir+"size")
    test_dataset = psf_dataset(root_dir=dataset_dir+"/", size=size,
                               transform=transforms.Compose([Normalize(), ToTensor()]),pretreatment=pretreatment)
    test_dataloader = DataLoader(test_dataset, batch_size=batchsize, num_workers=4, shuffle=False)
    for test_batch, test_batch_data in enumerate(test_dataloader):
        test_x = test_batch_data['image']
        label =test_batch_data['coff']
        test_y_pred = net(test_x)
        # print(test_y_pred.detach().numpy())
        # print(label.detach().numpy())

        phase_pred = np.squeeze(np.sum(test_y_pred.detach().numpy()[0, :, None, None] * zernike_basis[1:, :, :], axis=0))
        si_rmse = np.sqrt(mean_squared_error(phase_pred, test_batch_data['phase'].numpy()[0][0]))
        ci_rmse=np.sqrt(mean_squared_error(test_y_pred.detach().numpy(), label.detach().numpy()))
        print(si_rmse)

        rmse.append(si_rmse)
        c_rmse.append(ci_rmse)
        pre.append(test_y_pred.detach().numpy()[0])
        lable.append(label.detach().numpy()[0])
        phase_lable.append(test_batch_data['phase'].numpy()[0][0])
        phase_pre.append(phase_pred)
    print(eval_dir+"size: "+str(size))
    if os.path.exists(eval_dir)==False:
        os.makedirs(eval_dir)
    np.save(eval_dir+"/pre-" + althorm + ".npy", np.array(pre))
    np.save(eval_dir + "/pre-phase-" + althorm + ".npy", np.array(phase_pre))
    np.save(eval_dir+"/lable.npy", np.array(lable))
    np.save(eval_dir + "/lable-phase.npy", np.array(phase_lable))
    np.save(eval_dir+"/rmse-"+althorm +".npy",np.array(c_rmse))
    np.save(eval_dir+"/rmse-phase-"+althorm +".npy",np.array(rmse))



if __name__ == '__main__':
    # test()
    # if althorm=='our':
    #     print("phase-avg:", np.sum(np.load("./"+eval_dir+"/rmse-phase-" + althorm + loss + ".npy")) / 1000)
    print("avg:", np.sum(np.load("./"+eval_dir+"/rmse-" + althorm + loss + ".npy")) / 1000)
    # lable=np.load("./eval/lable-" + althorm + loss + ".npy")
    # pre_our=np.load("./eval/pre-" + althorm + ".npy")
    # print(len(lable))
    # print(len(pre_our))

