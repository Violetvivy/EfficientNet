import torch
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet

# 数据预处理 - 添加数据增强以减少过拟合
train_transform = transforms.Compose([
    transforms.Resize(640),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(640, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(640),
    transforms.CenterCrop(640),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = datasets.ImageFolder('data/dataset/train', transform=train_transform)
val_dataset = datasets.ImageFolder('data/dataset/val', transform=val_transform)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

# 加载预训练模型并修改输出类别数
model = EfficientNet.from_pretrained('efficientnet-b4', weights_path='efficientnet-b4-6ed6700e.pth', num_classes=len(train_dataset.classes))

# 损失函数和优化器 - 添加权重衰减作为正则化
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  # L2正则化

# 检查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

num_epochs = 50

# 跟踪最佳验证准确率和损失 - 添加早停机制
best_val_acc = 0.0
best_val_loss = float('inf')
best_epoch = 0
patience = 5  # 早停耐心值
patience_counter = 0
early_stop = False

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_loss = running_loss / total
    train_acc = correct / total

    # 验证
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_loss = val_loss / val_total
    val_acc = val_correct / val_total

    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")
    
    # 保存最佳模型权重（基于验证准确率）
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_loss = val_loss
        best_epoch = epoch + 1
        patience_counter = 0  # 重置耐心计数器
        torch.save(model.state_dict(), "efficientnet_b4_best.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            early_stop = True
    
    # 早停检查
    if early_stop:
        print(f"  训练提前停止在 epoch {epoch+1}")
        break

# 保存最终权重
torch.save(model.state_dict(), "efficientnet_b4_last.pth")
print(f"训练完成！")
print(f"最佳模型在 epoch {best_epoch}: 验证准确率 = {best_val_acc:.4f}, 验证损失 = {best_val_loss:.4f}")
