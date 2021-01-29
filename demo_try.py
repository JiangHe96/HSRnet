import argparse, os
import torch
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
from analysis import evaluate
from openpyxl import Workbook
from store2tiff import writeTiff as wtiff

test_ite=200
modelpath = 'checkpoint/HSInet_Block_Specg_SA_multiout_skip0.1/model_epoch_200.pth'
#image_input
data = io.loadmat('data/test.mat')
rgb = data['data']
rad=data['label']
img_rgb=rgb
img_refference=rad

#image_transpose
img_input = img_rgb.astype(float)
img_refference = img_refference.astype(float)
h = img_refference.shape[0]
w = img_refference.shape[1]
chanel = img_refference.shape[2]
img_input=np.transpose(img_input, [2,0,1])

#input_data_construction
input = img_input
input = Variable(torch.from_numpy(input).float()).view(1, -1, input.shape[1], input.shape[2])
index = np.zeros((5,chanel+1))
epoch=test_ite-1
#model_input
path = '%s%s%s' % ("checkpoint/", name, "/model_epoch_")
type=".pth"
parser = argparse.ArgumentParser(description="PyTorch HSRnet Demo")
parser.add_argument("--cuda", action="store_true", default=True, help="use cuda?")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--model", default=path+str(epoch+1)+type, type=str, help="model path")
opt = parser.parse_args()
cuda = opt.cuda
if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
model = torch.load(modelpath, map_location=lambda storage, loc: storage)["model"]
# model_forward
if cuda:
    model = model.cuda()
    input = input.cuda()
else:
    model = model.cpu()
start_time = time.time()
out = model(input)
out = out.cpu()
torch.cuda.empty_cache()
#save to .mat
elapsed_time = time.time() - start_time
output_temp = out.data[0].numpy().astype(np.float32)
output_temp=np.transpose(output_temp, [1,2,0])
output=output_temp
print("It takes {}s for processing".format(elapsed_time))

#analysis accuracy
index=evaluate.analysis_accu(img_refference,output,1)

#save index and output
outputpath='%s%d%s'%('output/',test_ite,'ite/')
analysispath='%s%d%s'%('analysis/',test_ite,'ite')
path = outputpath.strip()
filepath = path.rstrip("\\")
isExists = os.path.exists(filepath)
if not isExists:
    os.makedirs(filepath)
    print(filepath + ' 创建成功')
else:
    print(filepath + ' 目录已存在')

test_index=index
wb = Workbook()
ws = wb.active
ws.title="%d次迭代" %test_ite
x=test_index[0,:].tolist()
ws.append(x)
x=test_index[1,:].tolist()
ws.append(x)
x=test_index[2,:].tolist()
ws.append(x)
x=test_index[3,:].tolist()
ws.append(x)
x=test_index[4,:].tolist()
ws.append(x)
wb.save('%s%s' % (analysispath,'.xlsx'))
wtiff(output,output.shape[2],output.shape[0],output.shape[1],'%s%s%d%s' % (outputpath, 'output',test_ite,'.tiff'))
wtiff(img_refference,img_refference.shape[2],img_refference.shape[0],img_refference.shape[1],'%s%s' % (outputpath, 'refference.tiff'))

#show line
x=np.arange(1,32)
cc=index[0,1:32]
psnr=index[1,1:32]
ssim=index[2,1:32]
sam=np.repeat(index[3,0], 31, axis=0)
ergas=np.repeat(index[4,0], 31, axis=0)

plt.plot(x,cc,"b-",label="$CC$")
plt.title("CC")
plt.savefig('%s%s' % (outputpath, 'cc.png'),dpi=520)
plt.show()
plt.plot(x,psnr,"g-",label="$PSNR$")
plt.title("PSNR")
plt.savefig('%s%s' % (outputpath, 'psnr.png'),dpi=520)
plt.show()
plt.plot(x,ssim,"r-",label="$SSIM$")
plt.title("SSIM")
plt.savefig('%s%s' % (outputpath, 'ssim.png'),dpi=520)
plt.show()
plt.plot(x,sam,"y-",label="$SAM$")
plt.title("SAM")
plt.savefig('%s%s' % (outputpath, 'sam.png'),dpi=520)
plt.show()
plt.plot(x,ergas,"k-",label="$ERGAS$")
plt.title("ERGAS")
plt.savefig('%s%s' % (outputpath, 'ergas.png'),dpi=520)
plt.show()

fig,ax = plt.subplots(1, 3,figsize=(10, 4))
ax0=ax[0].imshow(img_refference[:, :, 10])
ax[0].set_title("GT(31bands)")
ax1=ax[1].imshow(img_rgb)
ax[1].set_title("Input(rgb)")
ax2=ax[2].imshow(abs(output[:, :, 10]-img_refference[:, :, 10]))
ax[2].set_title("Output_differ")
position=fig.add_axes([0.65, 0.06, 0.28, 0.03])#位置[左,下,右,上]
cb=plt.colorbar(ax2,cax=position,orientation='horizontal')#方向
plt.savefig('%s%s' % (outputpath, '目视对比.png'),dpi=520)
plt.show()
