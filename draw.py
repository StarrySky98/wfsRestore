import graph
import numpy as np
from phase3D2D import phase_graph_2D
from Zernike import zernike_data,compute_psf
import aotools


N = 128


def SR(phase,N):
    psf0 = compute_psf(np.zeros((N,N)))
    psf1 = compute_psf(phase)
    return np.max(psf1)/np.max(psf0)


def rms(phase):
    mean = np.mean(phase)
    phase_mean = np.ones((N, N)) * mean
    # phase_graph_2D(phase_mean, 'o.png')
    mask = aotools.circle(N/ 2, N)
    phase_mean = np.multiply(phase_mean, mask)
    # print(np.sum(np.square(phase - phase_mean))/(128*128))
    rms = np.sqrt(np.sum(np.square(phase - phase_mean)) /(N*N))
    # print(rms)
    return rms


def draw_Zernike_bar_05():
    g = graph.Graph()
    pre_our = np.load('./data-0.5-eval/pre-our-smoothL1.npy')
    pre_res = np.load('./data-0.5-eval/pre-ResNet50.npy')
    pre_ince = np.load('./data-0.5-eval/pre-InceptionV3.npy')
    lable = np.load('./data-0.5-eval/lable-our-smoothL1.npy')
    x_axi = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15',
             'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28',
             'C29', 'C30', 'C31', 'C32', 'C33', 'C34', 'C35']
    for i in range(10000):
        g.Zernike_Bar("./our-0.5-Zernike/"+str(i)+".png",x_axi,lable[i],pre_our[i])
        g.Zernike_all_Bar("./con-0.5-Zernike/" + str(i) + ".png", x_axi, lable[i], pre_our[i], pre_res[i], pre_ince[i])
        print(i)

def draw_Zernike_bar():
    g = graph.Graph()
    pre_our = np.load('./eval/pre-our-smoothL1.npy')
    print(pre_our[0],len(pre_our))

    pre_res = np.load('./eval/pre-ResNet50.npy')
    pre_ince = np.load('./eval/pre-InceptionV3.npy')
    lable = np.load('./eval/lable-our-smoothL1.npy')
    print(lable[0])
    x_axi = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15',
             'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28',
             'C29', 'C30', 'C31', 'C32', 'C33', 'C34', 'C35']
    for i in range(999):
        g.Zernike_Bar("./our-Zernike/"+str(i)+".png",x_axi,lable[i],pre_our[i])
        g.Zernike_all_Bar("./con-Zernike/" + str(i) + ".png", x_axi, lable[i], pre_our[i], pre_res[i], pre_ince[i])
        print(i)

def loss_contract():
    g = graph.Graph()
    savedir="./data-0.5-graph/train-loss-contract.png"
    legends=["RMSE","MAE","SmoothL1"]
    data_x = np.load('./data-0.5-graph/data-our-x.npy')
    data_y_rmse = np.load('./data-0.5-graph/data-our-y.npy')
    print(data_y_rmse.shape)
    data_y_mae = np.load('./data-0.5-graph/data-our-mae-y.npy')
    data_y_smoothL1=np.load('./data-0.5-graph/data-our-smoothL1-y.npy')
    yline=[]
    yline.append(data_y_rmse)
    yline.append(data_y_mae)
    yline.append(data_y_smoothL1)
    g.draw_multi_polyline(data_x,yline,savedir,legends,xlabel="epoch",ylabel=" ",title="train-loss-contract")


def draw_phase_05(althorm):
    N = 128
    zernike_basis = zernike_data(N)  # shape(36,N,N)
    if althorm == 'lable':
        pre = np.load('./data-0.5-eval/lable-phase.npy')
    elif althorm == 'our':
        pre = np.load('./data-0.5-eval/pre-' + althorm + '-smoothL1.npy')
    else:
        pre = np.load('./data-0.5-eval/pre-'+althorm+'.npy')
    print(pre[0].shape)
    print(zernike_basis.shape)
    for i in range(len(pre)):
        savedir = './data-0.5-phase/'+althorm+'/'+str(i)+'.png'
        if althorm == 'our':
            tmp = np.squeeze(np.sum(pre[i, :, None, None] * zernike_basis[1:, :, :], axis=0))
            phase_graph_2D(tmp,savedir)
        else:
            phase_graph_2D(pre[i], savedir)
        print(i)


def rms(phase):
    mean = np.mean(phase)
    phase_mean = np.ones((N, N)) * mean
    # phase_graph_2D(phase_mean, 'o.png')
    mask = aotools.circle(N/ 2, N)
    phase_mean = np.multiply(phase_mean, mask)
    # print(np.sum(np.square(phase - phase_mean))/(128*128))
    rms = np.sqrt(np.sum(np.square(phase - phase_mean)) /(N*N))
    # print(rms)
    return rms


def draw_rms_05():
    g = graph.Graph()
    zernike_basis = zernike_data(N)  # shape(36,N,N)

    pre_our = np.load("./data-0.5-eval/pre-our-smoothL1.npy")
    pre_Incep = np.load("./data-0.5-eval/pre-InceptionV3.npy")
    pre_Res = np.load("./data-0.5-eval/pre-ResNet50.npy")
    pre_Unet = np.load("./data-0.5-eval/pre-Unet.npy")
    pre_Unetpp = np.load("./data-0.5-eval/pre-Unetpp.npy")
    lable = np.load("./data-0.5-eval/lable-phase.npy")

    # print(lable[0], lable[0].shape)
    # print(rms(lable[0]))

    pre_our_phase = np.ones((len(pre_our), N, N))
    pre_Incep_phase = np.ones((len(pre_our), N, N))
    pre_Res_phase = np.ones((len(pre_our), N, N))
    print(pre_our_phase.shape)
    before_rms = []
    after_our_rms = []
    after_Incep_rms = []
    after_Res_rms = []
    after_Unet_rms = []
    after_Unetpp_rms = []

    for i in range(len(pre_our)):
        pre_our_phase[i] = np.squeeze(np.sum(pre_our[i, :, None, None] * zernike_basis[1:, :, :], axis=0))
        pre_Incep_phase[i] = np.squeeze(np.sum(pre_Incep[i, :, None, None] * zernike_basis[1:, :, :], axis=0))
        pre_Res_phase[i] = np.squeeze(np.sum(pre_Res[i, :, None, None] * zernike_basis[1:, :, :], axis=0))

    for i in range(len(pre_our)):
        before_rms.append(rms(lable[i]))
        after_our_rms.append(rms(lable[i]-pre_our_phase[i]))
        after_Incep_rms.append(rms(lable[i]-pre_Incep_phase[i]))
        after_Res_rms.append(rms(lable[i]-pre_Res_phase[i]))
        after_Unet_rms.append(rms(lable[i]-pre_Unet[i]))
        after_Unetpp_rms.append(rms(lable[i]-pre_Unetpp[i]))

    print("before-rms:", np.mean(before_rms))
    print("after-our-rms:", np.mean(after_our_rms))
    print("after-Incep-rms:", np.mean(after_Incep_rms))
    print("after-Res-rms:", np.mean(after_Res_rms))
    print("after-Unet-rms:", np.mean(after_Unet_rms))
    print("after-Unetpp-rms:", np.mean(after_Unetpp_rms))

    data_x = [i for i in range(1000)]
    yline=[]
    yline.append(before_rms)
    yline.append(after_our_rms)
    yline.append(after_Incep_rms)
    yline.append(after_Res_rms)
    yline.append(after_Unet_rms)
    yline.append(after_Unetpp_rms)
    # print(np.array(yline).shape)
    savedir = "./data-0.5-graph/rms-con.png"
    legends = ["before_rms", "after_our_rms", "after_Incep_rms", "after_Res_rms", "after_Unet_rms", "after_Unetpp_rms"]
    g.draw_multi_polyline(data_x, yline, savedir, legends, xlabel=" ", ylabel=" ", title="RMS-Before-After")


def draw_rms():
    g = graph.Graph()
    zernike_basis = zernike_data(N)  # shape(36,N,N)

    pre_our = np.load("./eval/pre-our-smoothL1.npy")
    pre_Incep = np.load("./eval/pre-InceptionV3.npy")
    pre_Res = np.load("./eval/pre-ResNet50.npy")
    pre_Unet = np.load("./eval/pre-Unet.npy")
    pre_Unetpp = np.load("./eval/pre-Unetpp.npy")
    lable = np.load("./eval/lable-phase.npy")

    # print(lable[0], lable[0].shape)
    # print(rms(lable[0]))

    pre_our_phase = np.ones((len(pre_our), N, N))
    pre_Incep_phase = np.ones((len(pre_our), N, N))
    pre_Res_phase = np.ones((len(pre_our), N, N))
    print(pre_our_phase.shape)
    before_rms = []
    after_our_rms = []
    after_Incep_rms = []
    after_Res_rms = []
    after_Unet_rms = []
    after_Unetpp_rms = []

    for i in range(len(pre_our)):
        pre_our_phase[i] = np.squeeze(np.sum(pre_our[i, :, None, None] * zernike_basis[1:, :, :], axis=0))
        pre_Incep_phase[i] = np.squeeze(np.sum(pre_Incep[i, :, None, None] * zernike_basis[1:, :, :], axis=0))
        pre_Res_phase[i] = np.squeeze(np.sum(pre_Res[i, :, None, None] * zernike_basis[1:, :, :], axis=0))

    for i in range(len(pre_our)):
        before_rms.append(rms(lable[i]))
        after_our_rms.append(rms(lable[i]-pre_our_phase[i]))
        after_Incep_rms.append(rms(lable[i]-pre_Incep_phase[i]))
        after_Res_rms.append(rms(lable[i]-pre_Res_phase[i]))
        after_Unet_rms.append(rms(lable[i]-pre_Unet[i]))
        after_Unetpp_rms.append(rms(lable[i]-pre_Unetpp[i]))

    print("before-rms:",np.mean(before_rms))
    print("after-our-rms:", np.mean(after_our_rms))
    print("after-Incep-rms:", np.mean(after_Incep_rms))
    print("after-Res-rms:", np.mean(after_Res_rms))
    print("after-Unet-rms:", np.mean(after_Unet_rms))
    print("after-Unetpp-rms:", np.mean(after_Unetpp_rms))


    data_x = [i for i in range(1000)]
    yline=[]
    yline.append(before_rms)
    yline.append(after_our_rms)
    yline.append(after_Incep_rms)
    yline.append(after_Res_rms)
    yline.append(after_Unet_rms)
    yline.append(after_Unetpp_rms)
    # print(np.array(yline).shape)
    savedir = "./graph/rms-con.png"
    legends = ["before_rms", "after_our_rms", "after_Incep_rms", "after_Res_rms", "after_Unet_rms", "after_Unetpp_rms"]
    g.draw_multi_polyline(data_x, yline, savedir, legends, xlabel=" ", ylabel=" ", title="RMS-Before-After")


if __name__ == '__main__':
    # loss_contract()
    draw_Zernike_bar_05()
    # draw_rms_05()
    # draw_phase_05('our')