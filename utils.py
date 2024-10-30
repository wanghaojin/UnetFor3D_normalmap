import torch
import torch.nn.functional as F
import numpy as np
def poisson_reconstruct(gx, gy):
    batch_size, H, W = gx.shape
    f = torch.zeros_like(gx)
    f[:, 1:-1, 1:-1] = (gx[:, 1:-1, 1:-1] - gx[:, 1:-1, :-2]) + (gy[:, 1:-1, 1:-1] - gy[:, :-2, 1:-1])
    f_hat = torch.fft.fft2(f)
    y_freq = torch.fft.fftfreq(H, d=1.0, device=gx.device).view(H, 1)
    x_freq = torch.fft.fftfreq(W, d=1.0, device=gx.device).view(1, W)

    denom = (2 * torch.cos(2 * np.pi * x_freq) - 2) + (2 * torch.cos(2 * np.pi * y_freq) - 2) 
    denom[0, 0] = 1 

    denom = denom.unsqueeze(0).to(gx.device) 
    denom = denom + 1e-8  

    Z = torch.real(torch.fft.ifft2(f_hat / denom))
    Z = Z - Z.view(batch_size, -1).min(dim=1)[0].view(batch_size, 1, 1)  

    return Z

def nonlinear_photometric_model(I, local_contrast, local_reflectance):
    contrast_factor = 1 + local_contrast
    reflectance_factor = 1 + local_reflectance
    I_nonlinear = torch.log1p(I)  # log(1 + I)
    I_adjusted = I_nonlinear * contrast_factor * reflectance_factor
    return I_adjusted

def photometric_stereo(I, L):
    batch_size, num_images, H, W = I.shape
    I = I.view(batch_size, num_images, -1)  # [batch_size, num_images, H*W]
    I = I.permute(0, 2, 1)  # [batch_size, H*W, num_images]

    # 应用非线性光度模型
    I = nonlinear_photometric_model(I, 0.3, 0.1)  # 参数可根据需要调整

    # 计算法线
    L_pinv = torch.pinverse(L)  # [3, num_images]
    N = torch.matmul(I, L_pinv.T)  # [batch_size, H*W, 3]

    # 对法线进行归一化
    N_norm = N.norm(dim=2, keepdim=True) + 1e-8
    N_normalized = N / N_norm  # [batch_size, H*W, 3]

    N_normalized = N_normalized.view(batch_size, H, W, 3)  # [batch_size, H, W, 3]
    return N_normalized  # 返回归一化的法线

def angular_loss(N_pred, N_label):
    # 确保法线不为零向量
    N_pred_norm = N_pred.norm(dim=-1, keepdim=True) + 1e-8
    N_label_norm = N_label.norm(dim=-1, keepdim=True) + 1e-8

    # 计算余弦相似度
    cos_theta = (N_pred / N_pred_norm) * (N_label / N_label_norm)
    cos_theta = cos_theta.sum(dim=-1)

    # 防止数值超出[-1, 1]
    cos_theta = torch.clamp(cos_theta, -1 + 1e-7, 1 - 1e-7)

    # 计算角度损失
    theta = torch.acos(cos_theta)
    loss = theta.mean()
    return loss
