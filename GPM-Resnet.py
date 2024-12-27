import torch
from torch import nn
from torch.nn.parameter import Parameter
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, models
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
import torch.nn as nn
import math
import torch.nn.functional as F

class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
class DropBlock(nn.Module):
    def __init__(self, drop_prob, block_size):
        super(DropBlock, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size
    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        else:
            gamma = self.drop_prob / (self.block_size ** 2)
            mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3]) < gamma).to(x.device)
            mask = mask.float()
            mask = -torch.nn.functional.max_pool2d(-mask, self.block_size, stride=1, padding=self.block_size // 2)
            mask = 1 - mask
            mask = mask / (1 - gamma)
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class ECABottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3,dropblock_prob=0.7, block_size=8):
        super(ECABottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)）
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.eca = eca_layer(planes * 4, k_size)
        self.downsample = downsample
        self.stride = stride
        self.dropblock_prob = dropblock_prob
        self.dropblock = DropBlock(dropblock_prob, block_size)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.dropblock_prob > 0.0:
            if self.training:
                out = self.dropblock(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.eca(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        self.dropblock(out)
        return out
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1, k_size=[3, 3, 3, 3],dropblock_prob=0.8, block_size=7):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], int(k_size[0]))
        self.layer2 = self._make_layer(block, 128, layers[1], int(k_size[1]), stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], int(k_size[2]), stride=2,dropblock_prob=dropblock_prob,
                                       block_size=block_size)
        self.layer4 = self._make_layer(block, 512, layers[3], int(k_size[3]), stride=2,dropblock_prob=dropblock_prob,
                                       block_size=block_size)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion,num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self, block, planes, blocks, k_size, stride=1, dropblock_prob=0.7, block_size=8):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, k_size, dropblock_prob, block_size))  # Pass dropblock_prob to the block
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
def GPM-ResNet(k_size=[3, 3, 3, 3], num_classes=1,dropblock_prob=0.8, block_size=7):
    model = ResNet(ECABottleneck, [3, 4, 6, 3], num_classes=num_classes, k_size=k_size,dropblock_prob=dropblock_prob, block_size=block_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model
model =  GPM-ResNet(k_size=[3, 3, 3, 3], num_classes=1,dropblock_prob=0.7, block_size=8)
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(in_features, 128),
    nn.ReLU(),
    nn.Linear(128, 1)

criterion = nn.MSELoss()

data_file = r'project/experiment/desi/image_h5_distribution/merged.h5'
print('3')
with h5py.File(data_file, 'r') as file:
    # 从文件中读取名为'oh_p50'的数据并存储在变量oh_p50中
    oh_p50 = file['oh_p50'][:]
    # 加载HDF5文件并读取除了'processed_images'之外的数据
    with h5py.File(data_file, 'r') as file:
        images = file['processed_images'][:]
        other_data = {}
        for key in file.keys():
            if key != 'processed_images':
                other_data[key] = file[key][:]
images = images.transpose(0, 2, 3, 1)
transform = transforms.Compose([
    transforms.ToTensor()
])
images = torch.stack([transform(img) for img in images])
label = torch.tensor(oh_p50, dtype=torch.float32)
train_images, remaining_images, train_targets, remaining_targets = train_test_split(images, label,
                                                                                    test_size=0.5,
                                                                                    random_state=123)

valid_images, test_images, valid_targets, test_targets = train_test_split(remaining_images, remaining_targets,
                                                                          test_size=0.4, random_state=123)
train_size = train_images.shape
train_count = len(train_targets)
print("训练集的尺寸 (Size)：", train_size, "训练集的数据数量 (Count)：", train_count)
valid_size = valid_images.shape
valid_count = len(valid_targets)
print("验证集的尺寸 (Size)：", valid_size, "验证集的数据数量 (Count)：", valid_count)
test_size = test_images.shape
test_count = len(test_targets)
print("测试集的尺寸 (Size)：", test_size, "测试集的数据数量 (Count)：", test_count)
# 创建PyTorch的TensorDataset，用于训练集和验证集
train_dataset = TensorDataset(train_images, train_targets)
valid_dataset = TensorDataset(valid_images, valid_targets)
test_dataset = TensorDataset(test_images, test_targets)
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
train_on_gpu = torch.cuda.is_available()
device = 'cuda' if train_on_gpu else 'cpu'
print('device:', device)
if train_on_gpu:
    model = model.to(device)
    print('CUDA is available! Training on GPU ...')
else:
    print('CUDA is not available. Training on CPU ...')
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
best_loss = np.inf
num_epochs = 50
save_path = 'project/detect/GPM-Resnet'
if not os.path.exists(save_path):
    os.makedirs(save_path)
def save_to_h5(data_images, data_targets, other_data, set_name, path):
    split_other_data = {key: value[:len(data_targets)] for key, value in other_data.items()}
    with h5py.File(f"{path}/{set_name}.h5", 'w') as h5f:
        h5f.create_dataset('images', data=data_images.numpy())
        for key, value in split_other_data.items():
            h5f.create_dataset(key, data=value)
save_to_h5(train_images, train_targets, other_data, 'train', save_path)
save_to_h5(valid_images, valid_targets, other_data, 'valid', save_path)
save_to_h5(test_images, test_targets, other_data, 'test', save_path)
save_dir_loss = "project/detect/GPM-Resnet/loss"
save_dir_train = "project/detect/GPM-Resnet/train"
save_dir_valid = "project/detect/GPM-Resnet/valid"
save_dir_test = "project/detect/GPM-Resnet/test"
save_dir_train_Histogram="project/detect/GPM-Resnet/train-Histogram"
save_dir_valid_Histogram="project/detect/GPM-Resnet/valid-Histogram"
save_dir_test_Histogram="project/detect/GPM-Resnet/test-Histogram"
if not os.path.exists(save_dir_loss):
    os.makedirs(save_dir_loss)
if not os.path.exists(save_dir_train):
    os.makedirs(save_dir_train)
if not os.path.exists(save_dir_valid):
    os.makedirs(save_dir_valid)
if not os.path.exists(save_dir_test):
    os.makedirs(save_dir_test)
if not os.path.exists(save_dir_train_Histogram):
    os.makedirs(save_dir_train_Histogram)
if not os.path.exists(save_dir_valid_Histogram):
    os.makedirs(save_dir_valid_Histogram)
if not os.path.exists(save_dir_test_Histogram):
    os.makedirs(save_dir_test_Histogram)
train_losses = []
valid_losses = []
train_mae_history = []
train_rmse_history = []
train_id_diff_history = []
valid_mae_history = []
valid_rmse_history = []
valid_id_diff_history = []

train_mae = 0.0
train_rmse = 0.0
train_id_diff = 0.0
valid_mae = 0.0
valid_rmse = 0.0
valid_id_diff = 0.0

for epoch in range(num_epochs):
    model.train()
    torch.cuda.empty_cache()
    train_mae = 0
    train_rmse = 0
    train_id_diff = 0
    predictions = []
    true_values = []
    train_diff = []
    train_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        train_loss += loss.item() * inputs.size(0)
        loss.backward()
        optimizer.step()
        outputs = outputs.to("cpu").data.numpy()
        targets = targets.to("cpu").data.numpy()
        predictions.extend(outputs.squeeze().flatten())  # 将预测值添加到列表中
        true_values.extend(targets.flatten())  # 将真实值添加到列表中

        xy = np.vstack([predictions, true_values])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        predictions_sorted = np.array(predictions)[idx]
        true_values_sorted = np.array(true_values)[idx]
        z_sorted = z[idx
        diff = np.array(true_values) - np.array(predictions)
        train_diff.extend(diff)
        train_mae += mean_absolute_error(true_values, predictions, multioutput='raw_values')
        train_rmse += np.sqrt(mean_squared_error(true_values, predictions, multioutput='raw_values'))
        train_id_diff += np.std(np.array(true_values) - np.array(predictions))

    scheduler.step()  #
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    train_mae /= len(train_loader)
    train_mae_history.append(train_mae)
    train_rmse /= len(train_loader)
    train_rmse_history.append(train_rmse)
    train_id_diff /= len(train_loader)
    train_id_diff_history.append(train_id_diff)
    torch.cuda.empty_cache()
    torch.save(model.state_dict(), project/detect/nn_GPM-Resnet_model.pt')
    model.eval()

    fig, axes = plt.subplots(figsize=(12, 8))
    axes.hist(predictions, bins='auto', color='white', alpha=0.75, histtype='step',
                    edgecolor='blue',linewidth=2)
    axes.hist(true_values, bins='auto', color='white', alpha=0.75, histtype='step',
                    edgecolor='red',linewidth=2
    axes.set_xlabel('Metallicity',fontsize=15)
    axes.set_ylabel('Number of galaxies',fontsize=15)
    legend = axes.legend(['predicted', 'true'], fontsize=15, loc='upper left')
    legend.get_texts()[0].set_color('blue')  # 设置'predicted'的标签颜色为蓝色
    legend.get_texts()[1].set_color('red')  # 设置'true'的标签颜色为红色
    image_filename = f"epoch_{epoch + 1}.png"
    image_path = os.path.join(save_dir_train_Histogram, image_filename)
    plt.savefig(image_path)
    print("训练集直方图图像保存成功")
    plt.show()
    plt.close()
    fig, axes = plt.subplots(figsize=(12, 10))
    scatter = axes.scatter(true_values_sorted,predictions_sorted,s=10, alpha=0.7, c=z_sorted, cmap='YlOrRd',
                          marker='o')
    cbar = plt.colorbar(scatter, label='Density', orientation='vertical', pad=0.010, ax=axes)
    axes.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], color='red',
              linestyle='--')
    distance_to_diagonal = 0.15
    axes.plot([min(true_values), max(true_values)],
              [min(true_values) + distance_to_diagonal, max(true_values) + distance_to_diagonal],
              color='blue', linestyle='--')
    axes.plot([min(true_values) + distance_to_diagonal, max(true_values) + distance_to_diagonal],
              [min(true_values), max(true_values)], color='blue', linestyle='--')
    axes.set_xlabel('True Labels')
    axes.set_ylabel('Predicted Values')
    axes.set_title('Scatter Plot of Train of oh_p50 [Epoch = ' + str(epoch + 1) + ']')
    axes.set_xlim(min(true_values), max(true_values))
    axes.set_ylim(min(true_values), max(true_values))
    inset_axes = axes.inset_axes([0.65, 0.05, 0.3, 0.3])
    inset_axes.hist(train_diff, bins='auto', color='white', alpha=0.75, histtype='step',
                    edgecolor='black')  # Set alpha and remove color
    inset_axes.set_xlabel('△Z')
    inset_axes.set_ylabel('number of galaxies')
    formatted_mae = np.array2string(train_mae, precision=4, separator=', ')
    axes.text(0.1, 0.9, f'MAE: {formatted_mae}', transform=axes.transAxes, fontsize=12,
              ha='center',
              va='center', bbox=dict(facecolor='white', alpha=0.7))
    formatted_rmse = np.array2string(train_rmse, precision=4, separator=', ')
    axes.text(0.1, 0.85, f' RMSE: {formatted_rmse}', transform=axes.transAxes, fontsize=12,
              ha='center',
              va='center', bbox=dict(facecolor='white', alpha=0.7))
    formatted_diff_std = np.array2string(train_id_diff, precision=4, separator=', ')
    axes.text(0.1, 0.8, f'diff_std: {formatted_diff_std}', transform=axes.transAxes, fontsize=12,
              ha='center',
              va='center', bbox=dict(facecolor='white', alpha=0.7))

    image_filename = f"epoch_{epoch + 1}.png"
    image_path = os.path.join(save_dir_train, image_filename)
    plt.savefig(image_path)
    print("训练集图像保存成功")
    plt.show()
    plt.close()

    valid_mae = 0
    valid_rmse = 0
    valid_id_diff = 0
    predictions_valid = []
    true_values_valid = []
    valid_diff = []
    valid_loss = 0.0
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            valid_loss += loss.item() * inputs.size(0)
            outputs = outputs.to("cpu").data.numpy()
            targets = targets.to("cpu").data.numpy()
            predictions_valid.extend(outputs.squeeze().flatten())  # 将预测值添加到列表中
            true_values_valid.extend(targets.flatten())  # 将真实值添加到列表中

            xy = np.vstack([predictions_valid, true_values_valid])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            predictions_sorted_valid = np.array(predictions_valid)[idx]
            true_values_sorted_valid = np.array(true_values_valid)[idx]
            z_sorted_valid = z[idx]
            diff = np.array(true_values_valid) - np.array(predictions_valid)
            valid_diff.extend(diff)
            valid_mae += mean_absolute_error(true_values_valid, predictions_valid, multioutput='raw_values')
            valid_rmse += np.sqrt(mean_squared_error(true_values_valid, predictions_valid, multioutput='raw_values'))
            valid_id_diff += np.std(np.array(true_values_valid) - np.array(predictions_valid))

    valid_loss /= len(valid_loader)
    valid_losses.append(valid_loss)
    valid_mae /= len(valid_loader)
    valid_mae_history.append(valid_mae)
    valid_rmse /= len(valid_loader)
    valid_rmse_history.append(valid_rmse)
    valid_id_diff /= len(valid_loader)
    valid_id_diff_history.append(valid_id_diff)

    fig, axes = plt.subplots(figsize=(12, 8)
    axes.hist(predictions_valid, bins='auto', color='white', alpha=0.75, histtype='step',
              edgecolor='blue', linewidth=2)
    axes.hist(true_values_valid, bins='auto', color='white', alpha=0.75, histtype='step',
              edgecolor='red', linewidth=2)
    axes.set_xlabel('Metallicity', fontsize=15)
    axes.set_ylabel('Number of galaxies', fontsize=15)
    legend = axes.legend(['predicted', 'true'], fontsize=15, loc='upper left')

    legend.get_texts()[0].set_color('blue')  # 设置'predicted'的标签颜色为蓝色
    legend.get_texts()[1].set_color('red')  # 设置'true'的标签颜色为红色

    image_filename = f"epoch_{epoch + 1}.png"
    image_path = os.path.join(save_dir_valid_Histogram, image_filename)
    plt.savefig(image_path)
    print("验证集直方图图像保存成功")
    plt.show()
    plt.close()
    fig, axes = plt.subplots(figsize=(12, 10))
    scatter = axes.scatter( true_values_sorted_valid,predictions_sorted_valid, s=10, alpha=0.7, c=z_sorted_valid,
                           cmap='YlOrRd',
                           marker='o')
    cbar = plt.colorbar(scatter, label='Density', orientation='vertical', pad=0.010, ax=axes)
    axes.plot([min(true_values_valid), max(true_values_valid)], [min(true_values_valid), max(true_values_valid)],
              color='red',
              linestyle='--')
    distance_to_diagonal = 0.15
    axes.plot([min(true_values_valid), max(true_values_valid)],
              [min(true_values_valid) + distance_to_diagonal, max(true_values_valid) + distance_to_diagonal],
              color='blue', linestyle='--')
    axes.plot([min(true_values_valid) + distance_to_diagonal, max(true_values_valid) + distance_to_diagonal],
              [min(true_values_valid), max(true_values_valid)], color='blue', linestyle='--')
    axes.set_xlabel('True Labels')
    axes.set_ylabel('Predicted Values')
    axes.set_title('Scatter Plot of Valid of oh_p50 [Epoch = ' + str(epoch + 1) + ']')
    axes.set_xlim(min(true_values_valid), max(true_values_valid))
    axes.set_ylim(min(true_values_valid), max(true_values_valid))
    inset_axes = axes.inset_axes([0.65, 0.05, 0.3, 0.3])
    inset_axes.hist(valid_diff, bins='auto', color='white', alpha=0.75, histtype='step',
                    edgecolor='black')  # Set alpha and remove color
    inset_axes.set_xlabel('△Z')
    inset_axes.set_ylabel('number of galaxies')
    formatted_mae = np.array2string(valid_mae, precision=4, separator=', ')
    axes.text(0.1, 0.9, f'MAE: {formatted_mae}', transform=axes.transAxes, fontsize=12,
              ha='center',
              va='center', bbox=dict(facecolor='white', alpha=0.7))
    formatted_rmse = np.array2string(valid_rmse, precision=4, separator=', ')
    axes.text(0.1, 0.85, f' RMSE: {formatted_rmse}', transform=axes.transAxes, fontsize=12,
              ha='center',
              va='center', bbox=dict(facecolor='white', alpha=0.7))
    formatted_diff_std = np.array2string(valid_id_diff, precision=4, separator=', ')
    axes.text(0.1, 0.8, f'diff_std: {formatted_diff_std}', transform=axes.transAxes, fontsize=12,
              ha='center',
              va='center', bbox=dict(facecolor='white', alpha=0.7))

    image_filename = f"epoch_{epoch + 1}.png"
    image_path = os.path.join(save_dir_valid, image_filename)
    plt.savefig(image_path)
    print("验证集保存成功")
    plt.show()
    plt.close()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
print(f"train_losses length: {len(train_losses)}")
print(f"valid_losses length: {len(valid_losses)}")
plt.plot(range(num_epochs), train_losses, label='Train Loss')
plt.plot(range(num_epochs), valid_losses, label='Valid Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.ylim(0, 20)
save_path = os.path.join(save_dir_loss, 'loss.png')
plt.savefig(save_path)
print("损失图保存成功")
plt.show()
plt.close()
model.eval()

test_mae_history = []
test_rmse_history = []
test_id_diff_history = []
test_nmad_history = []
test_rmse = 0
test_id_diff = 0
test_nmad = 0
predictions_test = []
true_values_test = []
test_diff = []
test_diff_abs=[]

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        outputs = outputs.to("cpu").data.numpy()
        targets = targets.to("cpu").data.numpy()
        predictions_test.extend(outputs.squeeze().flatten())  # 将预测值添加到列表中
        true_values_test.extend(targets.flatten())  # 将真实值添加到列表中

        xy = np.vstack([predictions_test, true_values_test])
        z = gaussian_kde(xy)(xy
        idx = z.argsort()
        predictions_sorted_test = np.array(predictions_test)[idx]
        true_values_sorted_test = np.array(true_values_test)[idx]
        z_sorted_test = z[idx]
        diff = np.array(predictions_test) - np.array(true_values_test)
        test_diff
        diff_abs = np.abs(np.array(predictions_test) - np.array(true_values_test))
        test_diff_abs.extend(diff_abs)
        xy_diff = np.vstack([np.array(true_values_test), diff])
        z_diff = gaussian_kde(xy_diff)(xy_diff)
        idx_diff = z_diff.argsort()
        true_values_sorted_test = np.array(true_values_test)[idx_diff]
        diff_sorted_test = np.array(diff)[idx_diff]
        z_sorted_test_diff = z_diff[idx_diff]
        test_mae += mean_absolute_error(predictions_test, true_values_test, multioutput='raw_values')
        test_id_diff += np.std(np.array(predictions_test) - np.array(true_values_test))
        test_rmse += np.sqrt(mean_squared_error(predictions_test, true_values_test, multioutput='raw_values'))

diff_mean = np.mean(test_diff)
median_test_diff = np.median(test_diff_abs)
median_diff = np.median(test_diff)
mad = np.median(np.abs(test_diff - median_diff))
nmad = 1.4826 * mad
test_nmad += nmad
test_mae /= len(test_loader)
test_mae_history.append(test_mae)
test_rmse /= len(test_loader)
test_rmse_history.append(test_rmse)
test_id_diff /= len(test_loader)
test_id_diff_history.append(test_id_diff)
test_nmad /= len(test_loader)
test_nmad_history.append(test_nmad)

print(f'mae的值为: {test_mae.item():.4f}')
print(f'rmse的值为: {test_rmse.item():.4f}')
print(f'标准差的值为: {test_id_diff.item():.4f}')
print(f'NMAD的值为: {test_nmad.item():.4f}')


interval_width = 0.2
interval_starts = np.arange(min(predictions_sorted_test), max(predictions_sorted_test), interval_width)
medians = []
std_devs_up = []
std_devs_down=[]# 新增一个列表，用于存储每个区间的残差标准差

for start in interval_starts:
    end = start + interval_width
    mask = (predictions_sorted_test >= start) & (predictions_sorted_test < end)
    predictions_in_interval = predictions_sorted_test[mask]
    true_values_in_interval = true_values_sorted_test[mask]  # 假设true_values_sorted_test是真实值的数组

    if len(predictions_in_interval) > 0:
        residuals= predictions_in_interval - true_values_in_interval
        median_in_interval = np.median(predictions_in_interval)
        residual_std = np.std(residuals)  # 计算残差的标准差
        residual_up=median_in_interval + residual_std
        residual_down = median_in_interval - residual_std
        medians.append(median_in_interval)
        std_devs_up.append( residual_up)
        std_devs_down.append(residual_down)
        # 将残差的标准差添加到列表中
    else:
        medians.append(np.nan)
        std_devs_up.append(np.nan)
        std_devs_down.append(np.nan)# 如果区间内没有数据，将NaN添加到残差标准差列表中，表示缺失值



fig, axes = plt.subplots(figsize=(10, 8))
axes.hist(predictions_test, bins='auto', color='white', alpha=0.75, histtype='step',
          edgecolor='blue', linewidth=2.5)
axes.hist(true_values_test, bins='auto', color='white', alpha=0.75, histtype='step',
          edgecolor='red', linewidth=2.5)
axes.set_xlabel('gas-phase metallicity', fontsize=20)
axes.set_ylabel('Number of galaxies', fontsize=20)
axes.tick_params(axis='x', labelsize=15)
axes.tick_params(axis='y', labelsize=15)
legend = axes.legend(['prediction','true'], fontsize=15, loc='upper left')
# 设置图例中标签的颜色
legend.get_texts()[0].set_color('blue')  # 设置'predicted'的标签颜色为蓝色
legend.get_texts()[1].set_color('red')  # 设置'true'的标签颜色为红色

image_filename = f"test_Histogram.png"
image_path = os.path.join(save_dir_test_Histogram, image_filename)
plt.savefig(image_path)
print("测试集直方图图像保存成功")
plt.close()

figs = plt.figure(figsize=(9, 12))
gs = figs.add_gridspec(2, 1, height_ratios=(1, 3), hspace=0.05, wspace=1)
ax_scatter = figs.add_subplot(gs[1])
scatter_t = ax_scatter.scatter(true_values_sorted_test, predictions_sorted_test,  s=10, alpha=0.7, c=z_sorted_test,
                       cmap='rainbow', marker='o')
cax_height = ax_scatter.get_position().height
cax = figs.add_axes([0.92, ax_scatter.get_position().y0, 0.02, cax_height])

cbar = figs.colorbar(scatter_t, orientation='vertical', cax=cax)
cbar.ax.tick_params(labelsize=15)
ax_scatter.plot([min(true_values_test), max(true_values_test)], [min(true_values_test), max(true_values_test)], color='black',
          linestyle='-')
ax_scatter.plot([min(true_values_test), max(true_values_test)], [min(true_values_test), max(true_values_test)], color='black',
          linestyle='-')
ax_scatter.plot(interval_starts + interval_width / 2, medians, color='red', linestyle='-', linewidth=2,
          label='Median Line')
ax_scatter.plot(interval_starts + interval_width / 2, std_devs_up, color='purple', linestyle='--', linewidth=2,
          label='Median Line')
ax_scatter.plot(interval_starts + interval_width / 2, std_devs_down, color='purple', linestyle='--', linewidth=2,
          label='Median Line')
ax_scatter.tick_params(axis='both', which='both', labelsize=15)
ax_scatter.set_xlabel('$Z_{\mathrm{true}}$',fontsize=24)
ax_scatter.set_ylabel('$Z_{\mathrm{pred}}$',fontsize=24)
ax_scatter.set_xlim(min(true_values_test), max(true_values_test))
ax_scatter.set_ylim(min(true_values_test), max(true_values_test))

inset_axes = ax_scatter.inset_axes([0.68, 0.08, 0.3, 0.3])
inset_axes.hist(test_diff, bins='auto', color='white', alpha=0.75, histtype='step',
                edgecolor='black')
inset_axes.tick_params(axis='both', which='both', labelsize=12)
inset_axes.set_xlabel(r'$\bigtriangleup Z$',fontsize=15)
inset_axes.set_ylabel('number of galaxies',fontsize=13)

formatted_mean = np.array2string(np.squeeze(diff_mean1), precision=4)
ax_scatter.text(0.12, 0.93, f'μ: {formatted_mean}', transform=ax_scatter.transAxes, fontsize=16,
          ha='center',
          va='center', bbox=dict(facecolor='none', edgecolor='none'))
formatted_diff_std = np.array2string(np.squeeze(test_id_diff), precision=4)
ax_scatter.text(0.12, 0.86, f'σ: {formatted_diff_std}', transform=ax_scatter.transAxes, fontsize=16,
          ha='center',
          va='center', bbox=dict(facecolor='none', edgecolor='none'))
formatted_nmad = np.array2string(np.squeeze(test_nmad), precision=4)
ax_scatter.text(0.12, 0.79, f'NMAD: {formatted_nmad}', transform=ax_scatter.transAxes, fontsize=16,
          ha='center',
          va='center', bbox=dict(facecolor='none', edgecolor='none'))

ax_histx = figs.add_subplot(gs[0], sharex=ax_scatter)
ax_histx.scatter(true_values_sorted_test,diff_sorted_test , s=10, alpha=0.7, c=z_sorted_test_diff,
                       cmap='rainbow', marker='o')
ax_histx.set_ylim(-1.2, 1.2)
ax_histx.set_yticks([-1, -0.5, 0, 0.5, 1])
ax_histx.tick_params(axis='x', which='both',  labelbottom=False)
ax_histx.tick_params(axis='both', which='both', labelsize=15)
ax_histx.axhline(0, color='red', linestyle='-', linewidth=2, label='y=0')
ax_histx.axhline(0.1, color='purple', linestyle='--', linewidth=2, label='y=0.1')
ax_histx.axhline(-0.1, color='purple', linestyle='--', linewidth=2, label='y=-0.1')
ax_histx.set_ylabel('$Z_{\mathrm{pred}}$-$Z_{\mathrm{true}}$',fontsize=22)
ax_histx.set_xlim(min(true_values_test), max(true_values_test))
plt.subplots_adjust(hspace=1)
plt.tight_layout()
figs.suptitle('GPM-ResNet',fontsize=30)
figs.subplots_adjust(top=0.93)

image_filename = f"test.png"
save_path = os.path.join(save_dir_test, 'GPM-ResNet.png')
plt.savefig(save_path)
print("测试集保存成功")
plt.show()
plt.close()