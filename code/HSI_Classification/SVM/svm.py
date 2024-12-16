# -*-coding=utf-8 -*-
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
import scipy.io as sio
import random
import matplotlib.pyplot as plt
import os

path = os.path.join( '/home/storage/Dataset/WHU-Hi-HanChuan')# change this path for your dataset
PaviaU = os.path.join(path,'WHU_Hi_HanChuan.mat')
PaviaU_gt = os.path.join(path,'WHU_Hi_HanChuan_gt.mat')
method_path = 'SVM'

# 加载数据
data = sio.loadmat(PaviaU)
print(data)
data_gt = sio.loadmat(PaviaU_gt)
print(data_gt)
im = data['WHU_Hi_HanChuan'] # pair with your dataset
imGIS = data_gt['WHU_Hi_HanChuan_gt']# pair with your dataset
# 归一化
im = (im - float(np.min(im)))
im = im/np.max(im)
print(im.shape)
imGIS =np.where(imGIS==255,0,imGIS)
#print(imGIS,np.max(imGIS))
sample_num =200
deepth = im.shape[2]
classes = np.max(imGIS)
test_bg = False

data_pos = {}
train_pos = {}
test_pos = {}

for i in range(1,classes+1):
    #print(i)
    data_pos[i]=[]
    train_pos[i]=[]
    test_pos[i] = []
#print(range(imGIS.shape[0]))
for i in range(imGIS.shape[0]-1):
    for j in range(imGIS.shape[1]-1):
        for k in range(1,classes+1):
            if imGIS[i,j]==k:
                data_pos[k].append([i,j])
                continue

for i in range(1,classes+1):
    print(range(len(data_pos[i])))
    indexies = random.sample(range(len(data_pos[i])),sample_num)
    for k in range(len(data_pos[i])):
        if k not in indexies:
            test_pos[i].append(data_pos[i][k])
        else:
            train_pos[i].append(data_pos[i][k])

train = []
train_label = []
test = []
test_label = []

for i in range(1,len(train_pos)+1):
    for j in range(len(train_pos[i])):
        row,col = train_pos[i][j]
        train.append(im[row,col])
        train_label.append(i)

for i in range(1,len(test_pos)+1):
    for j in range(len(test_pos[i])):
        row,col = test_pos[i][j]
        #print(row,col)
        test.append(im[row,col])
        test_label.append(i)
if not os.path.exists(os.path.join(method_path,'result2')):
    os.makedirs(os.path.join(method_path,'result2'))

clf = SVC(C=100,kernel='rbf',gamma=1)
train = np.asarray(train)
train_label = np.asarray(train_label)
clf.fit(train,train_label)
C = np.max(imGIS)

matrix = np.zeros((C,C))
for i in range(len(test)):
    r = clf.predict(test[i].reshape(-1,len(test[i])))
    matrix[r-1,test_label[i]-1] += 1

ac_list = []
for i in range(len(matrix)):
    ac = matrix[i, i] / sum(matrix[:, i])
    ac_list.append(ac)
    print(i+1,'class:','(', matrix[i, i], '/', sum(matrix[:, i]), ')', ac)
print('confusion matrix:')
print(np.int_(matrix))
print('total right num:', np.sum(np.trace(matrix)))
print('total test num:',np.sum(matrix))
accuracy = np.sum(np.trace(matrix)) / np.sum(matrix)
print('Overall accuracy:', accuracy)
# kappa
kk = 0
for i in range(matrix.shape[0]):
    kk += np.sum(matrix[i]) * np.sum(matrix[:, i])
pe = kk / (np.sum(matrix) * np.sum(matrix))
pa = np.trace(matrix) / np.sum(matrix)
kappa = (pa - pe) / (1 - pe)
ac_list = np.asarray(ac_list)
aa = np.mean(ac_list)
print('Average accuracy:',aa)
print('Kappa:', kappa)
sio.savemat(os.path.join('result', 'resultSWHU_Hi_HanChuan.mat'), {'oa': accuracy,'aa':aa,'kappa':kappa,'ac_list':ac_list,'matrix':matrix})

iG = np.zeros((imGIS.shape[0],imGIS.shape[1]))
for i in range(imGIS.shape[0]-1):
    for j in range(imGIS.shape[1]-1):
        if imGIS[i,j] == 0:
            if test_bg:
                iG[i,j] = (clf.predict(im[i,j].reshape(-1,len(im[i,j]))))
            else:
                iG[i,j]=0
        else:
            iG[i,j] = (clf.predict(im[i,j].reshape(-1,len(im[i,j]))))
if test_bg:
    iG[0,0] = 0

de_map = iG[::-1]
fig, _ = plt.subplots()
height, width = de_map.shape
fig.set_size_inches(width/100.0, height/100.0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
plt.axis('off')
plt.axis('equal')
plt.pcolor(de_map, cmap='jet')
plt.savefig(os.path.join('result', 'WHU_Hi_HanChuan_decode_map.png'),format='png',dpi=600)#bbox_inches='tight',pad_inches=0)
plt.close()
print('decode map get finished')

