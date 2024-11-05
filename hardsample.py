# 导入必要的库
from model import UNet
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# 数据集定义
class AugmentedDepthDataset(Dataset):
    def __init__(self, image_paths, depth_paths, transform=None):
        self.image_paths = image_paths
        self.depth_paths = depth_paths
        self.transform = transform

    def __len__(self):
        return len(self.depth_paths) * 4

    def __getitem__(self, idx):
        original_idx = idx // 4
        crop_idx = idx % 4
        rotation_angle = random.choice([0])

        images = []
        for img_path in self.image_paths[original_idx]:
            img = Image.open(img_path).convert('L')
            img = img.rotate(rotation_angle)
            img = img.resize((1024, 1024))

            if self.transform:
                img = self.transform(img)
            images.append(img)

        input_tensor = torch.cat(images, dim=0)  # [4, H, W]
        depth = np.load(self.depth_paths[original_idx]).astype(np.float32)  # [H, W]
        depth_img = Image.fromarray(depth)
        depth_img = depth_img.rotate(rotation_angle)
        depth_img = depth_img.resize((1024, 1024))
        depth = np.array(depth_img)
        depth = torch.from_numpy(depth).unsqueeze(0)  # [1, H, W]

        # 随机裁剪
        i, j, h, w = transforms.RandomCrop.get_params(
            depth_img, output_size=(512, 512))
        input_tensor = transforms.functional.crop(input_tensor, i, j, h, w)
        depth = transforms.functional.crop(depth, i, j, h, w)

        return input_tensor, depth

# 数据加载器定义
def get_data_loaders(dataset_dir, batch_size=24, num_workers=4):
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
                print(f"{group_num}深度图不存在。")
        else:
            print(f"{group_num}不足4张图像。")
    print(f"共找到 {count} 个数据集。")
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])

    dataset = AugmentedDepthDataset(
        valid_image_paths, valid_depth_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers)
    return dataloader

# 训练脚本
dataset_dir = 'dataset'
train_loader = get_data_loaders(dataset_dir, batch_size=16)

model = UNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.L1Loss(reduction='none')  # 不进行内部求平均
optimizer = optim.Adam(model.parameters(), lr=0.00001)
alpha = 0.7
num_epochs = 1000000
losses = []

for epoch in tqdm(range(num_epochs)):
    model.train()
    running_loss = 0.0
    for inputs, depths in train_loader:
        inputs = inputs.to(device)    # [batch_size, 4, 512, 512]
        depths = depths.to(device)    # [batch_size, 1, 512, 512]
        depth_gradient = torch.gradient(depths, dim=(2, 3))
        optimizer.zero_grad()
        outputs = model(inputs)
        output_gradient = torch.gradient(outputs, dim=(2, 3))
        pixel_errors = torch.abs(outputs - depths).detach()
        weights = (pixel_errors ** 2) + 1e-8  # 加上小常数防止除零
        weights = weights / torch.mean(weights)  # 归一化
        l1_loss = criterion(outputs, depths)  # [batch_size, 1, H, W]
        weighted_l1_loss = torch.mean(l1_loss * weights)
        loss = alpha * weighted_l1_loss 

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    losses.append(epoch_loss)

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    if (epoch+1) % 10 == 0:
        with open('loss.txt', 'a', encoding='utf-8') as file:
            file.write(
                f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}\n')

    if (epoch + 1) % 500 == 0:
        torch.save(model.state_dict(), f'model_arg/unet_epoch_{epoch+1}.pth')

# 绘制损失曲线
plt.figure()
plt.plot(range(1, num_epochs + 1), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.savefig('training_loss_plot.png')
plt.show()

print('训练完成')
