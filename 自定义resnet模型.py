import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms as tt
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

# 数据增强与预处理
stats = ((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025))
train_transform = tt.Compose([
    tt.RandomHorizontalFlip(),
    tt.RandomCrop(32, padding=4, padding_mode="reflect"),
    tt.ToTensor(),
    tt.Normalize(*stats)
])
test_transform = tt.Compose([
    tt.ToTensor(),
    tt.Normalize(*stats)
])

# 数据集
train_data = CIFAR10(download=True, root="./data", transform=train_transform)
test_data = CIFAR10(root="./data", train=False, transform=test_transform)

# 数据加载器
BATCH_SIZE = 128
train_dl = DataLoader(train_data, BATCH_SIZE, num_workers=4, pin_memory=True, shuffle=True)
test_dl = DataLoader(test_data, BATCH_SIZE, num_workers=4, pin_memory=True, shuffle=False)

# 设备选择
def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class ToDeviceLoader:
    def __init__(self, data, device):
        self.data = data
        self.device = device

    def __iter__(self):
        for batch in self.data:
            yield to_device(batch, self.device)

    def __len__(self):
        return len(self.data)

device = get_device()
train_dl = ToDeviceLoader(train_dl, device)
test_dl = ToDeviceLoader(test_dl, device)

# 计算准确率的函数
def accuracy(predicted, actual):
    _, predictions = torch.max(predicted, dim=1)
    return torch.tensor(torch.sum(predictions == actual).item() / len(predictions))

# 基础模型类
class BaseModel(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {"val_loss": loss.detach(), "val_acc": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        loss = torch.stack(batch_losses).mean()
        batch_accuracies = [x["val_acc"] for x in outputs]
        acc = torch.stack(batch_accuracies).mean()
        return {"val_loss": loss.item(), "val_acc": acc.item()}

    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch}], last_lr: {result['lrs'][-1]:.5f}, train_loss: {result['train_loss']:.4f}, "
              f"val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")

# 创建了一个简单的卷积快捷连接块，由一个卷积层和一个批归一化层组成
def conv_shortcut(in_channel, out_channel, stride):
    layers = [nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=(stride, stride)),
              nn.BatchNorm2d(out_channel)]
    return nn.Sequential(*layers)

# 构建了一个基本的残差块，包含了两个卷积层的序列
def block(in_channel, out_channel, k_size, stride, conv=False):
    first_layers = [nn.Conv2d(in_channel, out_channel[0], kernel_size=(1, 1), stride=(1, 1)),
                    nn.BatchNorm2d(out_channel[0]),
                    nn.ReLU(inplace=True)]
    if conv:
        first_layers[0].stride = (stride, stride)
    second_layers = [nn.Conv2d(out_channel[0], out_channel[1], kernel_size=(k_size, k_size), stride=(1, 1), padding=1),
                     nn.BatchNorm2d(out_channel[1])]
    layers = first_layers + second_layers
    return nn.Sequential(*layers)

class ResNet(BaseModel):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.stg1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # stage 2
        self.convShortcut2 = conv_shortcut(64, 256, 1)
        self.conv2 = block(64, [64, 256], 3, 1, conv=True)
        self.ident2 = block(256, [64, 256], 3, 1)

        # stage 3
        self.convShortcut3 = conv_shortcut(256, 512, 2)
        self.conv3 = block(256, [128, 512], 3, 2, conv=True)
        self.ident3 = block(512, [128, 512], 3, 1)

        # stage 4
        self.convShortcut4 = conv_shortcut(512, 1024, 2)
        self.conv4 = block(512, [256, 1024], 3, 2, conv=True)
        self.ident4 = block(1024, [256, 1024], 3, 1)

        # Classify
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, inputs):
        out = self.stg1(inputs)

        # stage 2
        out = F.relu(self.conv2(out) + self.convShortcut2(out))
        out = F.relu(self.ident2(out) + out)
        out = F.relu(self.ident2(out) + out)
        out = F.relu(self.ident2(out) + out)

        # stage 3
        out = F.relu(self.conv3(out) + self.convShortcut3(out))
        out = F.relu(self.ident3(out) + out)
        out = F.relu(self.ident3(out) + out)
        out = F.relu(self.ident3(out) + out)
        out = F.relu(self.ident3(out) + out)

        # stage 4
        out = F.relu(self.conv4(out) + self.convShortcut4(out))
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)

        # Classify
        out = self.classifier(out)
        return out

# 训练评估与拟合功能
@torch.no_grad()
def evaluate(model, test_dl):
    model.eval()
    outputs = [model.validation_step(batch) for batch in test_dl]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit(epochs, train_dl, test_dl, model, optimizer, scheduler, grad_clip=None):
    torch.cuda.empty_cache()
    history = []
    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        for batch in train_dl:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            lrs.append(get_lr(optimizer))
        scheduler.step()
        result = evaluate(model, test_dl)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history
%%time
# 设置参数
# 模型、优化器和学习率调度器的定义
model = ResNet(in_channels=3, num_classes=10).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_dl), epochs=50)

# 训练模型
history = fit(50, train_dl, test_dl, model, optimizer, scheduler)

### 获取最终准确率
def get_final_accuracy(model, test_dl):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_dl:
            images, labels = batch
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# 计算并打印最终准确率
final_accuracy = get_final_accuracy(model, test_dl)
print(f'Final Accuracy: {final_accuracy:.2f}%')
### 获取最终损失率
def get_final_loss(model, test_dl):
    model.eval()
    total_loss = 0
    total = 0
    with torch.no_grad():
        for batch in test_dl:
            images, labels = batch
            outputs = model(images)
            loss = nn.functional.cross_entropy(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
    return total_loss / total

# 计算并打印最终损失率
final_loss = get_final_loss(model, test_dl)
print(f'Final Loss: {final_loss:.2f}')
import numpy as np
import matplotlib.pyplot as plt
# 准确率，损失率，学习率可视化
def plot_acc(history):
    plt.plot([x["val_acc"] for x in history],"-x")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
def plot_loss(history):
    plt.plot([x.get("train_loss") for x in history], "-bx")
    plt.plot([x["val_loss"] for x in history],"-rx")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train loss","val loss"])
plot_loss(history)
# 对图像进行预测并可视化
def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return test_data.classes[preds[0].item()]

# 预测第1张图片并可视化
img, label = test_data[0]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_data.classes[label], ', Predicted:', predict_image(img, model))
# 预测第1005张图片并可视化
img, label = test_data[1005]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_data.classes[label], ', Predicted:', predict_image(img, model))