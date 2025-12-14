import torch
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
import sys


# 此处设置图片路径和模型权重路径
image_path = 'fair-apple5.jpg'  # 图片路径
model_path = 'efficientnet_b4_best.pth'  # 模型权重路径

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(640),
    transforms.CenterCrop(640),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载类别名
import os
train_dir = 'data/dataset/train'
class_names = sorted(os.listdir(train_dir))

# 加载模型
model = EfficientNet.from_name('efficientnet-b4', num_classes=len(class_names))
model.load_state_dict(torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
model.eval()

# 读取图片并预处理
img = Image.open(image_path).convert('RGB')
input_tensor = transform(img).unsqueeze(0)

# 推理
with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)
    pred_class = class_names[predicted.item()]

print(f"预测类别: {pred_class}")
