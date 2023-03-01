import os.path

from flask import Flask, request, jsonify
import graph
import test_Unet
import test as t
import train
import train_Unet
import numpy as np
import phase3D2D
import Zernike
import draw
import genpsf as g
import preprocess
app = Flask(__name__)

"""
    批量复原
"""
@app.route('/batchRestore', methods=["GET", "POST"])
def batch_restore():  # put application's code here
    pretreatment = request.form.get('pretreatment')
    test_count = request.form.get("test_count")
    test_path = request.form.get("test_path")
    althorm = request.form.get("althrom")
    ex_id = request.form.get("ex_id")
    mode = request.form.get("mode")
    model_dir = './model1-35-'
    eval_dir = "D://wfsExperiment//exper//"+ ex_id + '/eval/'

    # 加载模型测试
    if mode == "Phase":
        test_Unet.test(althorm=althorm, model_dir=model_dir, eval_dir=eval_dir, dataset_dir=test_path,
                       size=int(test_count), pretreatment=int(pretreatment))
    elif mode == "Zernike":
        t.model_test(althorm=althorm, model_dir=model_dir, eval_dir=eval_dir, dataset_dir=test_path,
                     size=int(test_count), pretreatment=int(pretreatment),N=128)
    else:
        pass
    if os.path.exists(eval_dir)==False:
        os.makedirs(eval_dir)
    phase_pre = np.load(eval_dir + "/pre-phase-" + althorm + ".npy")
    avg_phase_rmse = np.mean(np.load(eval_dir + "/rmse-phase-" + althorm + ".npy"))
    phase_lable = np.load(eval_dir + "/lable-phase.npy")

    before_phase_path_list = []
    before_psf_path_list = []
    after_phase_path_list = []
    after_psf_path_list = []

    web_before_phase_path_list = []
    web_before_psf_path_list = []
    web_after_phase_path_list = []
    web_after_psf_path_list = []

    if os.path.exists('D://wfsExperiment//exper//' + ex_id + '/before_phase/')==False:
        os.makedirs('D://wfsExperiment//exper//' + ex_id + '/before_phase/')
    if os.path.exists('D://wfsExperiment//exper//' + ex_id + '/before_psf/')==False:
        os.makedirs('D://wfsExperiment//exper//' + ex_id + '/before_psf/')
    if os.path.exists('D://wfsExperiment//exper//' + ex_id + '/after_phase/')==False:
        os.makedirs('D://wfsExperiment//exper//' + ex_id + '/after_phase/')
    if os.path.exists('D://wfsExperiment//exper//' + ex_id + '/after_psf/')==False:
        os.makedirs('D://wfsExperiment//exper//' + ex_id + '/after_psf/')
    # 画复原前后相位图和PSF
    for i in range(int(test_count)):
        before_phase_path = 'D://wfsExperiment//exper//' + ex_id + '/before_phase/'+str(i)+'.png'
        before_psf_path = 'D://wfsExperiment//exper//' + ex_id + '/before_psf/'+str(i)+'.png'
        after_phase_path = 'D://wfsExperiment//exper//' + ex_id + '/after_phase/'+str(i)+'.png'
        after_psf_path = 'D://wfsExperiment//exper//' + ex_id + '/after_psf/'+str(i)+'.png'

        web_before_phase_path = 'wfsExperiment//exper//'+ex_id + '/before_phase/' + str(i) + '.png'
        web_before_psf_path = 'wfsExperiment//exper//'+ex_id + '/before_psf/' + str(i) + '.png'
        web_after_phase_path = 'wfsExperiment//exper//'+ex_id + '/after_phase/' + str(i) + '.png'
        web_after_psf_path = 'wfsExperiment//exper//'+ex_id + '/after_psf/' + str(i) + '.png'

        print("phase: " + str(phase_lable[i].shape))
        phase3D2D.phase_graph_3D(phase_lable[i] - phase_pre[i], after_phase_path)
        phase3D2D.phase_graph_3D(phase_lable[i], before_phase_path)

        phase3D2D.phase_graph_3D(Zernike.compute_psf(phase_lable[i] - phase_pre[i]), after_psf_path)
        phase3D2D.phase_graph_3D(Zernike.compute_psf(phase_lable[i]), before_psf_path)
        before_phase_path_list.append(before_phase_path)
        before_psf_path_list.append(before_psf_path)
        after_phase_path_list.append(after_phase_path)
        after_psf_path_list.append(after_psf_path)
        web_before_phase_path_list.append(web_before_phase_path)
        web_before_psf_path_list.append(web_before_psf_path)
        web_after_phase_path_list.append(web_after_phase_path)
        web_after_psf_path_list.append(web_after_psf_path)

    # 复原前后rms和sr
    after_rms_list = []
    before_rms_list = []
    after_sr_list = []
    before_sr_list = []
    for i in range(int(test_count)):
        after_rms = draw.rms(phase_lable[i] - phase_pre[i])
        before_rms = draw.rms(phase_lable[i])
        after_sr = draw.SR(phase_lable[i] - phase_pre[i], len(phase_lable[i]))
        before_sr = draw.SR(phase_lable[i] - phase_pre[i], len(phase_lable[i]))
        after_rms_list.append(after_rms)
        before_rms_list.append(before_rms)
        after_sr_list.append(after_sr)
        before_sr_list.append(before_sr)

    print(avg_phase_rmse)
    res_data = {
        "avg_phase_rmse": float(avg_phase_rmse),
        "before_phase_path": web_before_phase_path_list,
        "before_psf_path": web_before_psf_path_list,
        "after_phase_path": web_after_phase_path_list,
        "after_psf_path": web_after_psf_path_list,
        "after_rms": after_rms_list,
        "before_rms": before_rms_list,
        "after_sr": after_sr_list,
        "before_sr": before_sr_list,

    }
    if mode == 'Zernike':
        x = np.load(eval_dir + "/pre-" + althorm + ".npy")
        y = np.load(eval_dir + "/lable.npy")
        g = graph.Graph()
        save_dir = "D://wfsExperiment//exper//"+ ex_id + '/zernike/'
        web_save_dir = "wfsExperiment//exper//" + ex_id + '/zernike/'
        if os.path.exists(save_dir) == False:
            os.makedirs(save_dir)
        x_axi = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15',
                 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28',
                 'C29', 'C30', 'C31', 'C32', 'C33', 'C34', 'C35']
        print(np.load(eval_dir + "/lable.npy").shape)
        zenike_list_path = []
        web_zenike_list_path = []
        for i in range(int(test_count)):
            g.Zernike_Bar(save_dir+str(i)+".png", x_axi, x[i], y[i])
            zenike_list_path.append(save_dir+str(i)+".png")
            web_zenike_list_path.append(web_save_dir + str(i) + ".png")
        res_data["zernike_path"] = web_zenike_list_path
        res_data["avg_rmse"] = float(np.mean(np.load(eval_dir + "/rmse-" + althorm + ".npy")))
    return jsonify(res_data)


"""
    单个复原
"""
@app.route('/singleRestore', methods=["GET", "POST"])
def single_restore():  # put application's code here
    pretreatment = request.form.get('pretreatment')
    test_count = request.form.get("test_count")
    test_path = request.form.get("test_path")
    althorm = request.form.get("althorm")

    ex_id = request.form.get("ex_id")
    mode = request.form.get("mode") 
    model_dir = 'model1-35-'
    eval_dir = test_path+ex_id+'/eval/'
    print( mode)
    # 加载模型测试
    if mode == "Phase":
        test_Unet.test(althorm=althorm, model_dir=model_dir, eval_dir=eval_dir, dataset_dir=test_path,
                   size=int(test_count), pretreatment=int(pretreatment))
    elif mode == "Zernike":
        t.model_test(althorm=althorm, model_dir=model_dir, eval_dir=eval_dir, dataset_dir=test_path,
                       size=int(test_count), pretreatment=int(pretreatment), N=128)
    else:
        pass
    if mode=="Phase":
        phase_pre = np.load(eval_dir + "/pre-phase-" + althorm + ".npy")[0]
        avg_phase_rmse = np.mean(np.load(eval_dir + "/rmse-phase-" + althorm + ".npy"))
        phase_lable = np.load(eval_dir + "/lable-phase.npy")[0]
    else:
        phase_pre = np.load(eval_dir+"/pre-phase-"+althorm+".npy")
        avg_phase_rmse = np.mean(np.load(eval_dir+"/rmse-phase-"+althorm+".npy"))
        phase_lable = np.load(eval_dir+"/lable-phase.npy")


    # 画复原前后相位图和PSF
    before_phase_path = test_path+ex_id+'/before_phase/0.png'
    before_psf_path = test_path+ex_id+'/before_psf/0.png'
    after_phase_path = test_path+ex_id+'/after_phase/0.png'
    after_psf_path = test_path+ex_id+'/after_psf/0.png'
    web_before_phase_path ='wfsExperiment/upload/'+ ex_id + '/before_phase/0.png'
    web_before_psf_path = 'wfsExperiment/upload/'+ex_id + '/before_psf/0.png'
    web_after_phase_path = 'wfsExperiment/upload/'+ex_id + '/after_phase/0.png'
    web_after_psf_path = 'wfsExperiment/upload/'+ex_id + '/after_psf/0.png'
    if os.path.exists(test_path+ex_id+'/before_phase') == False:
        os.makedirs(test_path+ex_id+'/before_phase')
    if os.path.exists(test_path + ex_id + '/before_psf')== False:
        os.makedirs(test_path + ex_id + '/before_psf')
    if os.path.exists(test_path + ex_id + '/after_phase')== False:
        os.makedirs(test_path + ex_id + '/after_phase')
    if os.path.exists(test_path + ex_id + '/after_psf')== False:
        os.makedirs(test_path + ex_id + '/after_psf')

    print(phase_lable.shape)

    phase3D2D.phase_graph_3D(phase_lable-phase_pre, after_phase_path)
    phase3D2D.phase_graph_3D(phase_lable, before_phase_path)
    phase3D2D.phase_graph_3D(Zernike.compute_psf(phase_lable-phase_pre), after_psf_path)
    phase3D2D.phase_graph_3D(Zernike.compute_psf(phase_lable), before_psf_path)

    # 复原前后rms和sr
    after_rms = draw.rms(phase_lable-phase_pre)
    before_rms = draw.rms(phase_lable)
    after_sr = draw.SR(phase_lable-phase_pre, len(phase_lable))
    before_sr = draw.SR(phase_lable-phase_pre, len(phase_lable))
    print(before_rms)
    res_data = {
        "avg_phase_rmse": float(avg_phase_rmse),
        "before_phase_path": web_before_phase_path,
        "before_psf_path": web_before_psf_path,
        "after_phase_path": web_after_phase_path,
        "after_psf_path": web_after_psf_path,
        "after_rms": float(after_rms),
        "before_rms": float(before_rms),
        "after_sr": float(after_sr),
        "before_sr": float(before_sr),

    }

    if mode == 'Zernike':
        g = graph.Graph()
        save_dir = test_path+ex_id+'/zernike/0.png'
        web_save_dir= 'wfsExperiment/upload/'+ex_id+'/zernike/0.png'
        if os.path.exists(test_path+ex_id+'/zernike') == False:
            os.makedirs(test_path + ex_id + '/zernike')
        x_axi = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15',
                 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28',
                 'C29', 'C30', 'C31', 'C32', 'C33', 'C34', 'C35']
        print(np.load(eval_dir + "/lable.npy").shape)
        g.Zernike_Bar(save_dir, x_axi, np.load(eval_dir+"/lable.npy")[0], np.load(eval_dir+"/pre-"+althorm+".npy")[0])
        res_data["zernike_path"] = web_save_dir
        res_data["avg_rmse"] = float(np.mean(np.load(eval_dir+"/rmse-"+althorm+".npy")))
    return jsonify(res_data)


"""
    生成数据集
"""
@app.route('/genPSF', methods=["GET", "POST"])
def gen_psf():  # put application's code here
    zernike_coff_list = request.form.get('zernikeCoffList')
    train_count = request.form.get('trainCount')
    test_count = request.form.get('testCount')
    data_set_path = request.form.get('dataSetPath')
    image_size = request.form.get("imageSize")
    defous = request.form.get("defous")
    coff_list = np.float32(zernike_coff_list.split("[")[1].split("]")[0].split(','))
    print(train_count)
    gen_psf = g.GenPSF(n_zernike=35, train_psfs=int(train_count),test_psfs=int(test_count), N=int(image_size),coff_range=coff_list, defous=float(defous))
    gen_psf.gen_save_psf(str(data_set_path))
    res_data = {"200":"生成成功"}
    return jsonify(res_data)


"""
    预处理
"""
@app.route('/preprocess', methods=["GET", "POST"])
def data_propeprocess():  # put application's code here
    file_path = request.form.get('filePath')
    print(file_path)
    image_in_path, image_in_normal_path, image_in_normal_sqrt_path, image_in_normal_log_path = preprocess.preprocess(file_path)
    res_data = {
        "image_in": image_in_path,
        "image_in_normal_path": image_in_normal_path,
        "image_in_normal_sqrt_path": image_in_normal_sqrt_path,
        "image_in_normal_log_path": image_in_normal_log_path,
    }
    return jsonify(res_data)


"""
    训练
"""
@app.route('/train', methods=["GET", "POST"])
def data_train():  # put application's code here
    pretreatment = request.form.get('pretreatment')
    batchsize = request.form.get('batchsize')
    epoch = request.form.get('epoch')
    lr = request.form.get('lr')
    network = request.form.get('netWork')
    loss = request.form.get('loss')
    mode = request.form.get('mode')
    data_dir = request.form.get('dataset_dir')+"train/"
    train_count = request.form.get('train_count')

    model_dir = data_dir+"model/"
    if os.path.exists(model_dir)==False:
        os.makedirs(model_dir)
    if mode == "Zernike":
        data_x, data_y = train.train(network=network, loss_name=loss, train_dir=data_dir,
                    model_dir=model_dir, size=int(train_count), pretreatment=int(pretreatment),
                    batchsize=int(batchsize), epochs=int(epoch), lr=float(lr))
    else:
        data_x, data_y = train_Unet.train(network=network, loss_name=loss, train_dir=data_dir,
                                          model_dir=model_dir, size=int(train_count), pretreatment=int(pretreatment),
                                          batchsize=int(batchsize), epochs=int(epoch), lr=float(lr))
    g=graph.Graph()
    savedir="D:/wfsExperiment/upload//show/loss.png"
    web_savedir = "D:/wfsExperiment/upload//show/loss.png"
    if os.path.exists(data_dir+"/show/")==False:
        os.makedirs(data_dir+"/show/")
    g.draw_polyline(data_x,data_y,savedir,"epoch","value","trian_loss")
    res_data = {
        "loss_path": web_savedir

    }
    return res_data


"""
    测试与springboot通信
"""
@app.route('/test', methods=["GET", "POST"])
def test():
    pretreatment = request.form.get("pretreatment")
    test_count = request.form.get("test_count")
    test_path = request.form.get("test_path")
    althorm = request.form.get("althrom")
    ex_id = request.form.get("ex_id")
    mode = request.form.get("mode")
    print(pretreatment,mode,test_count)
    res_data = {
        "avg_phase_rmse": "1.png",
        "before_phase_path": "1.png",
        "before_psf_path": "1.png",
        "after_phase_path": "1.png",
        "after_psf_path": "1.png",
        "after_rms": "1.png",
        "before_rms": "1.png",
        "after_sr": "1.png",
        "before_sr": "1.png",

    }
    return jsonify(res_data)

if __name__ == '__main__':
    app.run()
