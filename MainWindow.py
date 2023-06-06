import os
import pickle
import time
import math
import tkinter as tk
import cv2

import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from functools import cmp_to_key

from sklearn.cluster import KMeans
from sklearn.metrics import recall_score, precision_score, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from bof import get_image_features
from image import is_image
from sift_surf import features
from tfidf import calsingletfidf, caltfidf, calidf

block = 150

class MainWindow:

    def __init__(self):
        super().__init__()
        self.root = None
        self.title = None
        self.sift = None
        self.surf = None
        self.enor = None
        self.enormous = None
        self.model = None
        self.choose = None
        self.match = None
        self.test = None
        self.showpic = None
        self.img = None
        self.MAP = None
        self.result = None
        self.modelname = None
        self.show = None
        self.para = None
        self.nn = None
        self.img = None
        self.shownum = 5

    def train(self, sift, clust100):
        self.manage = cv2.xfeatures2d.SIFT_create() if sift else cv2.xfeatures2d.SURF_create()
        cluster = 100 if clust100 else 20

        # train

        ## 选择训练集
        train_path = filedialog.askdirectory()
        if train_path == "":
            return
        print("train path: ", train_path)
        self.result.config(text="")

        train_des = []

        ## 计算训练时间
        start = time.time()

        ## 遍历训练集
        self.train_pic = []
        for root, dirs, files in os.walk(train_path):
            for file in files:
                if file == "Thumbs.db":
                    continue
                path = root + "/" + file
                if not is_image(path):
                    continue
                ## 将路径上的图片存入 train_pic
                img = cv2.imread(path)
                self.train_pic.append(img)
                ## 使用 SIFT，SURF 提取特征
                des = features(path, self.manage)
                ## 将提取的特征放入 train_des 中
                train_des.append(des)

        train_des2 = train_des
        train_des = np.vstack(train_des)
        ## 使用 kmeans 对提取的特征进行聚类
        self.kmeans = KMeans(n_clusters=cluster, random_state=0)
        self.kmeans.fit(train_des)

        self.train_features = []
        self.train_label = []
        i = 0
        for root, dirs, files in os.walk(train_path):
            for file in files:
                if file == "Thumbs.db":
                    continue
                path = root + "/" + file
                if not is_image(path):
                    continue
                ## sift 提取特征而后特征归类
                feature_vector = get_image_features(self.kmeans, train_des2[i])
                ## 整理数据集内容
                self.train_features.append(feature_vector)

                self.train_label.append(file[:6])

                i = i + 1

        ## 对训练集特征进行tfidf运算，并转置
        self.idf = calidf(self.train_features, self.kmeans.n_clusters)
        self.train_features = caltfidf(self.train_features, self.kmeans.n_clusters, self.idf)
        self.train_features = np.vstack(self.train_features)

        ## 计算训练时间
        end = time.time()
        print("train time:", end - start)

        self.knn = KNeighborsClassifier(n_neighbors=12)
        self.knn.fit(self.train_features, self.train_label)
        self.ovr = OneVsRestClassifier(self.knn)
        self.ovr.fit(self.train_features, self.train_label)

        self.nn = NearestNeighbors(n_neighbors=len(self.train_pic)).fit(self.train_features)

        filename = str(int(time.time())) + ".pickle"
        with open(filename, "wb") as file:
            pickle.dump([sift, self.kmeans, self.idf, self.nn, self.knn, self.train_pic, self.train_label, self.ovr, self.train_features], file)
        self.modelname.config(text="model:" + filename)

    def load(self):

        file = filedialog.askopenfilename()
        if file == "":
            return
        file_name = os.path.basename(file)
        file_extension = os.path.splitext(file_name)[1]
        if not file_extension == '.pickle':
            return

        self.modelname.config(text="model:" + str(file).split('/')[-1])
        with open(file, "rb") as f:
            sift, self.kmeans, self.idf, self.nn, self.knn, self.train_pic, self.train_label, self.ovr, self.train_features = pickle.load(f)
            self.manage = cv2.xfeatures2d.SIFT_create() if sift else cv2.xfeatures2d.SURF_create()

    def drawROC(self, test_features, test_label):

        test_score = self.ovr.predict_proba(test_features)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        label2int = dict()
        int2label = dict()
        for label in test_label:
            if not label in label2int:
                int2label[len(label2int)] = label
                label2int[label] = len(label2int)
        for label in set(test_label):
            tmp = []
            for label1 in test_label:
                if label == label1:
                    tmp.append(True)
                else:
                    tmp.append(False)
            i = label2int[label]
            fpr[i], tpr[i], _ = roc_curve(tmp, test_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        colors = ['red', 'green', 'blue', 'orange', 'black', 'gray', 'purple', 'pink', 'brown', 'yellow', 'green', 'blue', 'orange', 'black']
        for i, color in zip(range(self.ovr.n_classes_), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of {} (AUC = {:.2f})'.format(int2label[i], roc_auc[i]))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for multiclass')
        plt.legend(loc="lower right")
        plt.show()

        return

    def eval(self):

        if self.nn is None:
            return
        path = filedialog.askdirectory()
        if path == "":
            return

        print("test path: ", path)

        test_features = []
        test_label = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file == "Thumbs.db":
                    continue
                path = root + "/" + file
                des = features(path, self.manage)
                feature_vector = get_image_features(self.kmeans, des)
                test_features.append(feature_vector)
                test_label.append(file[:6])

        ## 对测试集特征进行tfidf运算，并转置
        test_features = caltfidf(test_features, self.kmeans.n_clusters, self.idf)

        test_features = np.vstack(test_features)

        test_pred = self.knn.predict(test_features)

        right = {}
        cnt = {}
        for i in test_label:
            right[i] = 0
            cnt[i] = 0
        for i in range(0, len(test_pred)):
            id = test_label[i]
            add = 1 if test_pred[i] == test_label[i] else 0
            right[id] += add
            cnt[id] += 1
        MAP = 0
        for name in right:
            MAP += right[name] / cnt[name]
        MAP /= len(right)

        self.MAP = MAP * 100
        self.recall = recall_score(test_label, test_pred, average='macro', zero_division=1.0) * 100
        self.precision = precision_score(test_label, test_pred, average='macro', zero_division=1.0) * 100
        str = ""
        str += "recall: " + "{:.2f}".format(self.recall) + "\n"
        str += "precision: " + "{:.2f}".format(self.precision) + "\n"
        str += "MAP: " + "{:.2f}".format(self.MAP) + "\n"
        self.result.config(text=str)

        self.drawROC(test_features, test_label)

    def InitWindow(self):
        # 设置窗口标题
        self.root = tk.Tk()
        self.root.title("图像匹配")
        self.root.geometry("800x600+0+0")

        # 窗口标题
        self.title = tk.Label(self.root, text="chiking 的简单图像匹配器", font=("Arial", 20))
        self.title.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

        # 判断是否要训练大模型
        self.enor = tk.IntVar()
        self.enormous = tk.Checkbutton(self.root, text="是否使用大模型训练", variable=self.enor)
        self.enormous.place(relx=0.4, rely=0.2, anchor=tk.CENTER)

        # 使用sift训练新模型
        self.sift = tk.Button(self.root, text="SIFT", width=10, height=2, command=lambda : self.train(True, self.enor.get()))
        self.sift.place(relx=0.1, rely=0.2, anchor=tk.CENTER)

        # 使用surf训练新模型
        self.surf = tk.Button(self.root, text="SURF", width=10, height=2, command=lambda : self.train(False, self.enor.get()))
        self.surf.place(relx=0.25, rely=0.2, anchor=tk.CENTER)

        # 使用已经训练好的模型
        self.model = tk.Button(self.root, text="选择模型", width=10, height=2, command=self.load)
        self.model.place(relx=0.1, rely=0.35, anchor=tk.CENTER)

        # 显示被选择的模型
        self.modelname = tk.Label(self.root)
        self.modelname.place(relx=0.25, rely=0.35, anchor=tk.CENTER)

        # 选择将匹配的图片
        self.choose = tk.Button(self.root, text="选择图片", width=10, height=2, command=self.ImportImage)
        self.choose.place(relx=0.1, rely=0.5, anchor=tk.CENTER)

        # 对图片进行匹配
        self.match = tk.Button(self.root, text="图片匹配", width=10, height=2, command=self.compare)
        self.match.place(relx=0.1, rely=0.65, anchor=tk.CENTER)

        # 显示最有可能的 n 张图片
        self.txt = tk.Label(self.root, text='显示的图片数量：')
        self.txt.place(relx=0.1, rely=0.8, anchor=tk.CENTER)
        self.getnum = tk.Entry(self.root)
        self.getnum.place(relx=0.25, rely=0.8, anchor=tk.CENTER)

        # 测试按钮
        self.test = tk.Button(self.root, text="模型测试", width=10, height=2, command=self.eval)
        self.test.place(relx=0.9, rely=0.2, anchor=tk.CENTER)

        # 显示被选择图片
        self.picshow = tk.Label(self.root, width=300, height=300)
        self.picshow.place(relx=0.3, rely=0.3)

        # 显示测试结果
        self.result = tk.Label(self.root, width=15, height=10, anchor='w')
        self.result.place(relx=0.9, rely=0.8, anchor=tk.CENTER)

        self.root.mainloop()

    def ImportImage(self):

        file_path = filedialog.askopenfilename()
        if file_path == "" or not is_image(file_path):
            return
        self.img = cv2.imread(file_path)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(self.img)
        img.thumbnail((block, block))
        img = ImageTk.PhotoImage(img)
        self.picshow.configure(image=img)
        self.picshow.image = img

    def labelshow(self, img, id):
        img = self.img if img == -1 else cv2.cvtColor(self.train_pic[img], cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img.thumbnail((block, block))
        img = ImageTk.PhotoImage(img)
        self.show[id].config(image=img)
        self.show[id].image = img

    def parashow(self, dis, id):
        self.para[id].config(text='dis:'+str(round(dis, 3)))
        return

    def clickleft(self):
        self.page -= 1
        if self.page == 1:
            self.left.config(state='disabled')
        if self.page == math.ceil(self.shownum / 5) - 1:
            self.right.config(state='normal')
        for i in range(5):
            self.show[i + 1].config(state=tk.NORMAL)
            self.labelshow(self.ids[0][self.showlist[i + (self.page - 1) * 5]], i + 1)
            self.parashow(self.dis[0][self.showlist[i + (self.page - 1) * 5]], i + 1)
        return

    def clickright(self):
        self.page += 1
        if self.page == math.ceil(self.shownum / 5):
            self.right.config(state='disabled')
        if self.page == 2:
            self.left.config(state='normal')
        show = 5 if self.page * 5 <= self.shownum else (self.shownum - 1) % 5 + 1
        for i in range(show):
            self.show[i + 1].config(state=tk.NORMAL)
            self.labelshow(self.ids[0][self.showlist[i + (self.page - 1) * 5]], i + 1)
            self.parashow(self.dis[0][self.showlist[i + (self.page - 1) * 5]], i + 1)
        for i in range(show, 5):
            self.show[i + 1].config(state=tk.DISABLED)
            self.para[i + 1].config(text="")
        return

    def compare(self):

        if self.img is None or self.nn is None:
            return
        if self.getnum.get().isdigit():
            self.shownum = min(int(self.getnum.get()), len(self.train_pic), 100)
        if self.shownum < 5:
            self.shownum = 5

        _, des = self.manage.detectAndCompute(self.img, None)
        vector = get_image_features(self.kmeans, des)
        vector = calsingletfidf(vector, self.kmeans.n_clusters, self.idf)
        # 由高到低返回某张图片的 距离以及编号
        self.dis, self.ids = self.nn.kneighbors(np.array(vector).reshape(1, -1))

        nvector = []
        for i in range(len(vector)):
            nvector.append(0)
        for i in range(self.shownum):
            for j in range(len(vector)):
                nvector[j] += self.train_features[self.ids[0][i]][j]
        for i in range(len(vector)):
            nvector[i] /= self.shownum
        vector = nvector
        self.dis, self.ids = self.nn.kneighbors(np.array(vector).reshape(1, -1))

        label = []
        # 将每个被显示的图像的 label 加入 label list中
        for i in range(self.shownum):
            label.append(self.train_label[self.ids[0][i]])
        # dis 代表被显示的图像的距离 ids 代表被显示图像在训练集中的编号
        dis = self.dis[0][ : int((self.shownum - 1) / 5 * 5 + 5)]
        ids = self.ids[0][ : int((self.shownum - 1) / 5 * 5 + 5)]
        # dic 代表不同的 label 对应的度量值
        dic = {}
        for i in range(self.shownum):
            if not dic.get(label[i]):
                dic[label[i]] = 0
            dic[label[i]] += 0.01 / (dis[i] + 0.01)

        # 经过排序之后的label list，按照度量值高到低不重复地展示不同的label
        def compare(a, b):
            if dic[a] > dic[b]:
                return -1
            elif dic[a] == dic[b]:
                return 0
            else:
                return 1
        label = sorted(list(set(label)), key=cmp_to_key(compare))

        # 在 showlist 中加入多个二元组
        # 二元组的两个元素分别为在 self.dis 中的索引编号
        self.showlist = []
        for name in label:
            for i, j in enumerate(ids):
                if self.train_label[j] == name:
                    self.showlist.append(i)

        root = tk.Toplevel(self.root)
        root.title("匹配结果")
        root.geometry("1000x500+0+0")
        root.resizable(False, False)

        # 布置显示框
        self.show = []
        self.para = []
        for i in range(6):
            self.show.append(None)
            self.para.append(None)
        self.show[0] = tk.Label(root, width=block, height=block)
        self.show[0].place(relx=0.5, y=130, anchor=tk.CENTER)
        for i in range(1, 6):
            self.show[i] = tk.Label(root, width=block, height=block)
            self.show[i].place(relx=-0.1+0.2*i, y=350, anchor=tk.CENTER)
            self.para[i] = tk.Label(root)
            self.para[i].place(relx=-0.1+0.2*i, y=250, anchor=tk.CENTER)

        # 内容显示
        self.labelshow(-1, 0)
        for i in range(5):
            self.labelshow(self.ids[0][self.showlist[i]], i + 1)
            self.parashow(self.dis[0][self.showlist[i]], i + 1)
        for i in range(min(self.shownum, 5), 5):
            self.show[i + 1].config(state=tk.DISABLED)
            self.para[i + 1].config(text='')

        # 布置左右键
        self.page = 1
        self.left = tk.Button(root, text='\u25C0', font=('Arial', 16), command=self.clickleft, state='disabled')
        self.left.place(relx=0.3, y=450)
        self.right = tk.Button(root, text='\u25B6', font=('Arial', 16), command=self.clickright)
        self.right.place(relx=0.7, y=450)
        if self.page == math.ceil(self.shownum / 5):
            self.right.config(state='disabled')

        root.mainloop()