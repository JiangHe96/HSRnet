import torch
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
import os
import seaborn as sns

path = 'E:/Spectral SR/Unet/analysis'
dirs = os.listdir(path)
count=0
index_temp=[]
name=[]
for i in dirs:
    if os.path.splitext(i)[1] == ".xlsx":
        wb = openpyxl.load_workbook('%s%s%s'%(path,'/',i))
        ws = wb.active
        data = np.zeros((5, 32))
        for r in range(1,6):
            for c in range(1, 33):
                data[r-1,c-1]=ws.cell(row=r,column=c).value
        index_temp.append(data)
        name.append(i.strip(os.path.splitext(i)[1]))

num=len(index_temp)
index=np.zeros((5,32,num))
for i in range(num):
    index[:,:,i]=index_temp[i]
    index[3, :, i] = np.repeat(index[3, 0, i], 32, axis=0)
    index[4, :, i] = np.repeat(index[4, 0, i], 32, axis=0)

x=np.arange(1,32)
figure=np.zeros((num,31))

sns.set_context("notebook")
for i in range(num):
    figure[i,:] = index[0, 1:32, i]
    plt.plot(x,figure[i],label=name[i])
plt.legend(loc="lower right")
plt.title("CC")
plt.savefig('%s%s%s'%(path,'/', 'cc.png'))
plt.show()

for i in range(num):
    figure[i,:] = index[1, 1:32, i]
    plt.plot(x,figure[i],label=name[i])
plt.legend(loc="lower right")
plt.title("PSNR")
plt.savefig('%s%s%s'%(path,'/', 'psnr.png'))
plt.show()

for i in range(num):
    figure[i,:] = index[2, 1:32, i]
    plt.plot(x,figure[i],label=name[i])
plt.legend(loc="lower right")
plt.title("SSIM")
plt.savefig('%s%s%s'%(path,'/', 'ssim.png'))
plt.show()

for i in range(num):
    figure[i,:] = index[3, 1:32, i]
    plt.plot(x,figure[i],label=name[i])
plt.legend(loc="lower right")
plt.title("SAM")
plt.savefig('%s%s%s'%(path,'/', 'sam.png'))
plt.show()

for i in range(num):
    figure[i,:] = index[4, 1:32, i]
    plt.plot(x,figure[i],label=name[i])
plt.legend(loc="lower right")
plt.title("ERGAS")
plt.savefig('%s%s%s'%(path,'/', 'ergas.png'))
plt.show()
