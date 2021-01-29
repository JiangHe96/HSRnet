import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.HSI_Renet_rgb import  HSI_ReNet_g
from dataset import DatasetFromHdf5
import matplotlib.pyplot as plt
import time
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description="PyTorch Sen2hyperion")
parser.add_argument("--batchSize", type=int, default=8, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=200, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate. Default=0.1")
parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_true",default=True, help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.1, help="Clipping Gradients. Default=0.4")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.5 ** (epoch // opt.step))
    return lr

def train(training_data_loader, optimizer, model, criterion, epoch,loss_vaule):
    lr = adjust_learning_rate(optimizer, epoch-1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()
    starttime=0
    for iteration, batch in enumerate(training_data_loader, 1):
        endtime = time.time()
        print(endtime - starttime)
        starttime =time.time()
        input,target = Variable(batch[0]), Variable(batch[1])
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), opt.clip)
        optimizer.step()
        print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.data))
        loss_vaule.append(loss.data)

def save_checkpoint(model, epoch,file):
    model_out_path = '%s%s%d%s' % (file, "model_epoch_",epoch,".pth")
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists(file):
        os.makedirs(file)

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))


#main fuction
opt = parser.parse_args()
print(opt)

cuda = opt.cuda
if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

opt.seed = random.randint(1, 10000)
print("Random Seed: ", opt.seed)
torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

cudnn.benchmark = True

print("===> Loading datasets")
train_set = DatasetFromHdf5("data/train26.h5")
filepath="checkpoint/CAVE/HSRnet_rgb_lr"+str(opt.lr)+"/"
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

print("===> Building model")
model = HSI_ReNet_g(31)
criterion = nn.L1Loss(size_average=True)
s = sum([np.prod(list(p.size())) for p in model.parameters()])
print ('Number of params: %d' % s)

print("===> Setting GPU")
if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

    # optionally resume from a checkpoint
if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        opt.start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"].state_dict())
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
if opt.pretrained:
    if os.path.isfile(opt.pretrained):
        print("=> loading model '{}'".format(opt.pretrained))
        weights = torch.load(opt.pretrained)
        model.load_state_dict(weights['model'].state_dict())
    else:
        print("=> no model found at '{}'".format(opt.pretrained))

print("===> Setting Optimizer")
opt_Adam = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay)
opt_SGD = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
opt_RMSprop= optim.RMSprop(model.parameters(), lr=opt.lr, alpha=0.9, weight_decay=opt.weight_decay)
optimizer=opt_Adam

print("===> Training")
loss=[]
for epoch in range(opt.start_epoch, opt.nEpochs + 1):
    train(training_data_loader, optimizer, model, criterion, epoch,loss)
    save_checkpoint(model, epoch,filepath)
x = range(10000, len(loss)+1)
plt.plot(x, loss[9999:len(loss)], '-')
plt.savefig('%s%s' % (filepath, 'loss.png'))
plt.show()