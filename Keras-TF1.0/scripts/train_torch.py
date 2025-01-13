# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision import models
from torch.utils.data import DataLoader

dir(models)



data_dir = './alexnet_mask/fake_face_data/dataset'

# Applying Transforms to the Data
image_transforms = { 
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=100, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=100),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=100),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}



# Load the Data
 
# Set train and valid directory paths
train_directory = './alexnet_mask/fake_face_data/dataset_splitted/train'
test_directory = './alexnet_mask/fake_face_data/dataset_splitted/test'

# No of epochs
epochs = 10

# Batch size
bs = 16
 
# Number of classes
num_classes = 2
 
# Load Data from folders
data = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    # 'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
    'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
}
 
# Size of Data, to be used for calculating Average Loss and Accuracy
train_data_size = len(data['train'])
# valid_data_size = len(data['valid'])
test_data_size = len(data['test'])
 
# Create iterators for the Data loaded using DataLoader module
trainloader = DataLoader(data['train'], batch_size=bs, shuffle=True)
# valid_data = DataLoader(data['valid'], batch_size=bs, shuffle=True)
testloader = DataLoader(data['test'], batch_size=bs, shuffle=True)


print(trainloader.dataset.classes)

device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")

model = models.resnet50(pretrained=True)
print('model summary', model)

for param in model.parameters():
    param.requires_grad = False
    
model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 10),
                                 nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
model.to(device)

epochs = 20
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses, train_acc, test_acc = [], [], [], []
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.train()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    acc = torch.mean(equals.type(torch.FloatTensor)).item()
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/print_every)
            test_losses.append(test_loss/len(testloader))          
            train_acc.append(acc)
            test_acc.append(accuracy/len(testloader))
            running_loss = 0
            model.train()
    print("Epoch:", (epoch+1),", Train loss: ",(running_loss/print_every),", Test Loss :", test_loss/len(testloader),", Train accuracy: ",acc, ", Test Accuracy: ", accuracy/len(testloader))
    
torch.save(model, 'aerialmodel.pth')

plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')

plt.plot(train_acc, label='Training Accuracy')
plt.plot(test_acc, label='Validation Accuracy')

plt.legend(frameon=False)
plt.show()
plot.savefig('plot_torch.png')