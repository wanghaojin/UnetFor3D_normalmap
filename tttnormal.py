import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from torchvision import transforms
import torch.nn.functional as F
from normal_m import UNet

def angular_loss(N_pred, N_label):
    # N_pred 和 N_label 的形状均为 [batch_size, H, W, 3]
    cos_theta = F.cosine_similarity(N_pred, N_label, dim=-1)
    loss = torch.acos(torch.clamp(cos_theta, -1 + 1e-7, 1 - 1e-7))
    return torch.mean(loss)

class AugmentedDepthDataset(Dataset):
    def __init__(self, image_paths, depth_paths):
        self.image_paths = image_paths
        self.depth_paths = depth_paths
        self.transform_variations = 4  # 每个样本的变换次数

    def __len__(self):
        return len(self.depth_paths) * self.transform_variations

    def __getitem__(self, idx):
        original_idx = idx // self.transform_variations

        # 随机旋转角度
        rotation_angle = random.choice([0, 90, 180, 270])

        # 裁剪尺寸
        crop_size = (512, 512)

        images = []
        i = j = h = w = None

        for idx_img, img_path in enumerate(self.image_paths[original_idx]):
            img = Image.open(img_path).convert('L')
            img = img.resize((1024, 1024))

            # 旋转
            img = img.rotate(rotation_angle, resample=Image.BILINEAR)

            # 获取裁剪参数
            if idx_img == 0:
                i, j, h, w = transforms.RandomCrop.get_params(
                    img, output_size=crop_size)

            # 裁剪
            img = transforms.functional.crop(img, i, j, h, w)

            # 转换为Tensor，不进行归一化
            img = self.pil_image_to_tensor_no_normalization(img)
            images.append(img)

        input_tensor = torch.cat(images, dim=0)  # [4, H, W]

        # 处理深度图
        depth = np.load(self.depth_paths[original_idx]).astype(np.float32)
        depth_img = Image.fromarray(depth)
        depth_img = depth_img.resize((1024, 1024))

        # 旋转
        depth_img = depth_img.rotate(rotation_angle, resample=Image.BILINEAR)

        # 裁剪
        depth_img = transforms.functional.crop(depth_img, i, j, h, w)

        # 转换为Tensor，不进行归一化
        depth = self.pil_image_to_tensor_no_normalization(depth_img)
        depth = depth.squeeze(0)  # 去除通道维度

        # 从深度图计算法向量
        N_label = self.compute_normal_map_from_depth(depth)

        return input_tensor, N_label

    def pil_image_to_tensor_no_normalization(self, img):
        np_img = np.array(img, dtype=np.float32)
        tensor_img = torch.from_numpy(np_img)
        if tensor_img.ndim == 2:
            tensor_img = tensor_img.unsqueeze(0)
        else:
            tensor_img = tensor_img.permute(2, 0, 1)
        return tensor_img  # 不进行归一化

    def compute_normal_map_from_depth(self, depth):
        # depth: [H, W]
        sobel_kernel_x = torch.tensor(
            [[[[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]]]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor(
            [[[[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]]]], dtype=torch.float32)

        # 边缘填充
        depth_padded = F.pad(depth.unsqueeze(0).unsqueeze(0),
                             (1, 1, 1, 1), mode='replicate')

        gx = F.conv2d(depth_padded, sobel_kernel_x)
        gy = F.conv2d(depth_padded, sobel_kernel_y)

        gx = gx.squeeze(0).squeeze(0)  # [H, W]
        gy = gy.squeeze(0).squeeze(0)

        ones = torch.ones_like(depth)

        N = torch.stack((-gx, -gy, ones), dim=0)  # [3, H, W]
        # N = F.normalize(N, p=2, dim=0)  # 对法向量进行归一化

        return N  # [3, H, W]

def get_data_loaders(dataset_dir, batch_size=8, num_workers=4):
    image_dir = os.path.join(dataset_dir, 'image')
    depth_dir = os.path.join(dataset_dir, 'depth')
    image_files = glob.glob(os.path.join(image_dir, '*.png'))
    group_images = defaultdict(list)

    for img_file in image_files:
        filename = os.path.basename(img_file)
        group_num, angle = filename.split('.')[0].split('-')
        group_images[group_num].append(img_file)

    valid_image_paths = []
    valid_depth_paths = []
    count = 0
    for group_num, img_list in group_images.items():
        if len(img_list) == 4:
            depth_file = os.path.join(depth_dir, f'{group_num}.npy')
            if os.path.exists(depth_file):
                img_list.sort(key=lambda x: int(
                    os.path.basename(x).split('-')[1].split('.')[0]))
                valid_image_paths.append(img_list)
                valid_depth_paths.append(depth_file)
                count += 1
            else:
                print(f"{group_num} 深度图不存在。")
        else:
            print(f"{group_num} 图片数量不足 4 张。")
    print(f"共计 {count} 个数据集。")

    dataset = AugmentedDepthDataset(valid_image_paths, valid_depth_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers)
    return dataloader

if __name__ == "__main__":
    dataset_dir = 'dataset'  # 请确保数据集目录正确
    train_loader = get_data_loaders(dataset_dir, batch_size=12)

    model = UNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    num_epochs = 1000000
    losses = []

    torch.autograd.set_detect_anomaly(True)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        for inputs, N_label in train_loader:
            inputs = inputs.to(device)      # [batch_size, 4, H, W]
            N_label = N_label.to(device)    # [batch_size, 3, H, W]

            optimizer.zero_grad()
            outputs = model(inputs)         # [batch_size, 3, H, W]

            # 调整维度并归一化
            N_pred = outputs.permute(0, 2, 3, 1)  # [batch_size, H, W, 3]
            # N_pred = F.normalize(N_pred, p=2, dim=-1)

            N_label = N_label.permute(0, 2, 3, 1)  # [batch_size, H, W, 3]

            # 计算角度损失
            loss = angular_loss(N_pred, N_label)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        losses.append(epoch_loss)
        if (epoch+1) % 10 == 0:
            with open('loss.txt', 'a', encoding='utf-8') as file:
                file.write(f' {epoch+1}/{num_epochs} epoch loss: {epoch_loss:.4f}\n')
        if (epoch + 1) % 500 == 0:
            torch.save(model.state_dict(),
                       f'model/unet_epoch_{epoch+1}.pth')

    plt.figure()
    plt.plot(range(1, num_epochs + 1), losses, label='loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss over epochs')
    plt.legend()
    plt.savefig('training_loss_plot.png')
    plt.show()

    print('训练完成')
