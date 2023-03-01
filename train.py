import torch
from vggnet import VGG
from dataset import psf_dataset, Normalize, ToTensor
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from InceptionV3 import Net
from resnet50 import ResNet50

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(network, loss_name, train_dir, model_dir, size,
          pretreatment=1, batchsize=32, epochs=161, lr=0.0001):
    if network == 'our':
        net = VGG("VGG16").cuda()
    elif network == 'ResNet50':
        net = ResNet50().cuda()
    else:
        net = Net().cuda()


    data_x = []
    data_y = []
    dataset = psf_dataset(root_dir=train_dir, size=size,
                          transform=transforms.Compose([Normalize(), ToTensor()]),
                          pretreatment=pretreatment)
    train_dataloader = DataLoader(dataset, batch_size=batchsize, num_workers=4, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    if loss_name == 'smoothL1':
        loss_func = torch.nn.SmoothL1Loss()
    elif loss_name == 'mae':
        loss_func = torch.nn.L1Loss()
    else:
        loss_func = torch.nn.MSELoss()

    for epoch in range(epochs):
        for i_batch, batch_data in enumerate(train_dataloader):
            if epoch % 40 == 0 and i_batch == 400:
                torch.save(net.state_dict(), model_dir+"/model-" + str(epoch) + ".pkl")
            x, y_true = batch_data['image'].cuda(), batch_data['coff'].cuda()
            y_pred = net(x)
            if loss_name == 'mae' or loss_name == 'smoothL1':
                loss = loss_func(y_pred, y_true)
            else:
                loss = torch.sqrt(loss_func(y_pred, y_true))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i_batch % 100 == 0:
                data_x.append(epoch)
                data_y.append(loss)
                print('epoch {}, iter{}, loss {:1.4f}'.format(epoch, i_batch, loss))

    return data_x, data_y


# if __name__ == '__main__':
#     train
