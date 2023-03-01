import matplotlib.pyplot as plt
import numpy as np


class Graph(object):
    """docstring for Graph"""
    def __init__(self):
        super(Graph, self).__init__()

        # 设置图例并且设置图例的字体及大小
        self.font = {
                'weight': 'normal',
                'size': 20,
                }
        self.font1 = {
                 'weight': 'normal',
                 'size': 20,
                 }
        self.line_type = ['-y','-b','-r','-g']


    def draw_polyline(self,x,y,savedir,xlabel="",ylabel="",title=""):
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.cla()
        # 设置输出的图片大小
        figsize = 6, 5
        plt.figure(figsize=figsize)

        # 设置坐标刻度值的大小以及刻度值的字体
        plt.tick_params(labelsize=25)
        plt.plot(x, y,linewidth=2)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(savedir, bbox_inches="tight")
        plt.close()


    def draw_multi_polyline(self,xline,ylines,savedir,legends,xlabel="",ylabel="",title=""):
        """
        :param xline:X周数据
        :param ylines: Y轴数据
        :param savedir:
        :param legends:
        :param xlabel:
        :param ylabel:
        :param title:
        :param types:
        :return:
        """
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.cla()

        plt.figure()

        # 设置坐标刻度值的大小以及刻度值的字体
        # plt.tick_params(labelsize=25)


        for i in range(len(ylines)):
            plt.plot(xline, ylines[i], label=legends[i])
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(savedir)
        plt.close()

    def compare_MSE(self,xline,ylines,savedir,legends,xlabel="",ylabel="",title=""):
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.cla()
        # 设置输出的图片大小
        figsize = 6, 5
        plt.figure(figsize=figsize)

        # 设置坐标刻度值的大小以及刻度值的字体
        colors = ["midnightblue","skyblue","saddlebrown","sandybrown","darkgreen","limegreen"]
        zorders = [1,0,2]
        plt.tick_params(labelsize=25)
        types = [':', '--', '-', '-.']
        for i in range(len(ylines)):
            plt.plot(xline, ylines[i][2],types[i], label=legends[i],zorder = zorders[i])
            plt.fill_between(xline, ylines[i][0], ylines[i][1],facecolor=colors[i*2+1],alpha=0.5,zorder = zorders[i])
        plt.legend(prop=self.font1)
        plt.xlabel(xlabel,self.font)
        plt.ylabel(ylabel,self.font)
        plt.title(title)
        plt.savefig(savedir, bbox_inches="tight")
        plt.close()

    def compare_one_menchanism(self, savedir, workersNum, xlabel, ylabel, param):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        bar_width = 0.3  # 条形宽度
        index_our_mechanism = np.arange(len(workersNum))  # 男生条形图的横坐标
        # 使用两次 bar 函数画出两组条形图
        plt.bar(index_our_mechanism, height=param, width=bar_width, color='b', label='DDM mechanism')
        if ylabel == 'diversity Num':
            plt.ylim(0, 10)
        plt.legend(loc='best')  # 显示图例
        plt.ylabel(ylabel)  # 纵坐标轴标题
        plt.xlabel(xlabel)  # 图形标题
        plt.xticks(index_our_mechanism, workersNum)
        plt.savefig(savedir, bbox_inches="tight")
        plt.close()

    def Zernike_Bar(self, savedir, x_axi, truth, prediction):
        """

        :param x_axi: X轴坐标
        :param bar_y1: Y轴坐标
        :param bar_y2: Y轴坐标
        :return:
        """
        bar_width=0.35
        x = np.arange(len(x_axi))  # the label locations
        plt.figure(figsize=(15, 5), dpi=80)
        # 画柱状图，width可以设置柱子的宽度
        plt.bar(x, truth, width=bar_width,align="center",color="c",label="truth")
        plt.bar(x+bar_width, prediction, width=bar_width,align="center",color="b",label="prediction")

        plt.legend()  # 显示图例
        plt.ylabel('value')  # 纵坐标轴标题
        plt.xticks(x+bar_width/2, x_axi)
        plt.title('Zernike Cofficient')
        # plt.show()
        plt.savefig(savedir)
        plt.close()

    def Zernike_all_Bar(self, savedir, x_axi, truth, our_prediction, Res_prediction,Incep_prediction):
        """

        :param x_axi: X轴坐标
        :param bar_y1: Y轴坐标
        :param bar_y2: Y轴坐标
        :return:
        """
        bar_width=0.2
        x = np.arange(len(x_axi))  # the label locations
        plt.figure(figsize=(20, 5), dpi=80)
        # 画柱状图，width可以设置柱子的宽度
        plt.bar(x, truth, width=bar_width,align="center",label="truth")
        plt.bar(x+bar_width, our_prediction, width=bar_width,align="center",label="ours")
        plt.bar(x + 2*bar_width, Res_prediction, width=bar_width, align="center",  label="ResNet50")
        plt.bar(x + 3 * bar_width, Incep_prediction, width=bar_width, align="center",  label="InceptionV3")
        plt.legend()  # 显示图例
        plt.ylabel('value')  # 纵坐标轴标题
        plt.xticks(x+bar_width/2, x_axi)
        plt.title('Zernike Cofficient')
        # plt.show()
        plt.savefig(savedir)
        plt.close()

    def compare_boxs(self, datas, savedir, labels, xlabel="", ylabel="", title=""):
        """

        :param datas: 数据
        :param savedir: 存储地址
        :param labels: 横轴坐标
        :param xlabel: 横轴意义
        :param ylabel: 纵轴意义
        :param title: 标题
        :return:
        """
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.cla()
        # 设置输出的图片大小
        figsize = 10, 5
        plt.figure(figsize=figsize)
        # 设置坐标刻度值的大小以及刻度值的字体
        plt.tick_params(labelsize=10)

        plt.boxplot(datas, labels=labels, medianprops={'color': 'red', 'linewidth': '1.5'},
                meanline=True,
                showmeans=True,
                meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
                flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},showfliers=False)
        plt.xlabel(xlabel, self.font)
        plt.ylabel(ylabel, self.font)
        plt.title(title)
        plt.savefig(savedir, bbox_inches="tight")
        plt.close()



if __name__ == '__main__':
    # labels = ['G1', 'G2', 'G3', 'G4', 'G5']
    # men_means = [20, 34, 30, 35, 27]
    # women_means = [25, 32, 34, 20, 25]
    # y = [10,23,78,47,90]
    # z = [10,23,78,47,90]
    # x =[10,23,78,47,90]
    # savedir="./pics/test4.jpg"
    g=Graph()
    # g.Zernike_all_Bar(savedir,labels,men_means,women_means,y,x)
    labels = 'A', 'B', 'C', 'D', 'E', 'F'
    A = [0.4978, 0.5764, 0.5073, 0.5609]
    B = [0.5996, 0.65, 0.6251, 0.6473]
    C = [0.6015, 0.687, 0.6237, 0.6761]
    D = [0.5918, 0.6999, 0.6343, 0.6947]
    E = [0.577, 0.6932, 0.6593, 0.7036]
    F = [0.5637, 0.7161, 0.6683, 0.697]
    plt.grid(True)  # 显示网格
    plt.boxplot([A, B, C, D, E, F],
                medianprops={'color': 'red', 'linewidth': '1.5'},
                meanline=True,
                showmeans=True,
                meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
                flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},
                labels=labels)
    plt.yticks(np.arange(0.4, 0.81, 0.1))
    plt.show()

