import torch


class SAM(torch.nn.Module):
    def __init__(self, size_average = False):
        super(SAM, self).__init__()

    def forward(self, img_base, img_out):
        sum1 = torch.sum(img_base * img_out, 1)
        sum2 = torch.sum(img_base * img_base, 1)
        sum3 = torch.sum(img_out * img_out, 1)
        t = (sum2 * sum3) ** 0.5
        numlocal = torch.gt(t, 0)
        num = torch.sum(numlocal)
        t = sum1 / t
        angle = torch.acos(t)
        sumangle = torch.where(torch.isnan(angle), torch.full_like(angle, 0), angle).sum()
        if num == 0:
            averangle = sumangle
        else:
            averangle = sumangle / num
        SAM = averangle * 180 / 3.14159256
        return SAM
