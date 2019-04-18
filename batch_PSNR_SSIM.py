import numpy
import argparse
import scipy.misc
import os.path

from TorchTools.Functions.SRMeasure import psnr, ssim_exact, niqe

parser = argparse.ArgumentParser()
parser.add_argument('--gt', type=str, default='', help='Ground Truth image path')
parser.add_argument('--img', type=str, default='', help='Image dir path')
opt = parser.parse_args()

abs_img_path = os.path.abspath(opt.img)
img_list = os.listdir(abs_img_path)
abs_img_list = []

for i in range(len(img_list)):
    abs_img_list.append(os.path.join(abs_img_path, img_list[i]))

gt = os.path.abspath(opt.gt)
ref = scipy.misc.imread(gt, flatten=True).astype(numpy.float32)

PSNR_list = []
SSIM_list = []
NIQE_list = []

return_str = 'GT: %s\n' % gt
for i in range(len(img_list)):
    return_str += '%d. %s: ' % (i + 1, img_list[i])
    img = scipy.misc.imread(abs_img_list[i], flatten=True).astype(numpy.float32)
    img_psnr = psnr(img, ref)
    img_ssim = ssim_exact(img, ref)
    img_niqe = niqe(img)
    PSNR_list.append(img_psnr)
    SSIM_list.append(img_ssim)
    NIQE_list.append(img_niqe)
    print('%d. %s: PSNR: %.4f, SSIM: %.4f, NIQE: %.4f' % (i + 1, img_list[i], img_psnr, img_ssim, img_niqe))
    return_str += 'PSNR: %.4f, SSIM: %.4f, NIQE: %.4f\n' % (img_psnr, img_ssim, img_niqe)
PSNR = sum(PSNR_list) / len(PSNR_list)
SSIM = sum(SSIM_list) / len(SSIM_list)
NIQE = sum(NIQE_list) / len(NIQE_list)
return_str += 'AVG: PSNR: %.4f, SSIM: %.4f, NIQE: %.4f' % (PSNR, SSIM, NIQE)

with open('test_%s.txt' % opt.gt, 'r') as f:
    f.write(return_str)

