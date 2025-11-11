import torch
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = datasets.ImageFolder('data/dataset/train', transform=transform)
val_dataset = datasets.ImageFolder('data/dataset/val', transform=transform)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# 加载预训练模型并修改输出类别数
model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=len(train_dataset.classes))
