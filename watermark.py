#Test For Push

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image

# تنظیم دستگاه (GPU یا CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# تنظیمات
img_size = 224
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# تغییر اندازه و نرمال‌سازی تصاویر
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# بارگذاری داده‌های آموزشی
train_dir = 'path_to_train_folder'
train_data = ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# تعریف مدل CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)  # تخت کردن خروجی برای لایه Fully Connected
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# ساخت مدل و تعریف تابع هزینه و بهینه‌سازی‌گر
model = CNN().to(device)
criterion = nn.BCELoss()  
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# آموزش مدل
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))  # تطابق ابعاد
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# پیش‌بینی روی داده‌های آزمون
test_dir = 'path_to_test_folder'
test_images = os.listdir(test_dir)
predictions = []

model.eval()
with torch.no_grad():
    for img_name in test_images:
        img_path = os.path.join(test_dir, img_name)
        img = Image.open(img_path)
        img = transform(img).unsqueeze(0).to(device)  # تغییر اندازه و اضافه کردن بعد batch
        output = model(img)
        pred = 1 if output.item() > 0.5 else 0
        predictions.append((img_name, pred))

#ذخیره نتایج  
df = pd.DataFrame(predictions, columns=['image_name', 'predicted'])
df.to_csv('output.csv', index=False)


import zipfile
with zipfile.ZipFile('output.zip', 'w') as zf:
    zf.write('output.csv')
