import torch
import sys
from torch import nn
import os
from tqdm import tqdm

torch.manual_seed(3407)
sys.path.append('../')


def Finetune(train_loader, val_loader, model_checkpoints_folder, device, encoder, epochs=100):
    optimizer = torch.optim.SGD(encoder.parameters(), lr=0.3, nesterov=True, momentum=0.9, weight_decay=0.0001)

    # Define your loss function (e.g., CrossEntropyLoss, MSE, etc.)
    criterion = nn.CrossEntropyLoss()
    # for name, param in encoder.named_parameters():
    #   if "projetion" in name:
    #       param.requires_grad = False
    # resnet_example = models.resnet18(pretrained=False)
    # encoder = torch.nn.Sequential(*list(encoder.children())[:-1])
    # new_fc_layer = nn.Linear(128, 3)
    # encoder.add_module("new_fc", new_fc_layer)
    encoder.to(device)
    encoder.train()  # Set the model to training mode
    best_loss = float('inf')
    open(os.path.join(model_checkpoints_folder, 'checkpoints/loss_fine.txt'), 'w').close()
    for epoch in range(epochs):
        # 训练阶段
        encoder.train()
        train_loss = 0.0
        for i, (x, y) in enumerate(tqdm(train_loader)):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            features = encoder(x)
            loss = criterion(features, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        encoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)
                features = encoder(x)
                loss = criterion(features, y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch + 1}/{epochs}]\tTraining Loss: {avg_train_loss:.4f}\tValidation Loss: {avg_val_loss:.4f}")

        with open(os.path.join(model_checkpoints_folder, 'checkpoints/loss_fine.txt'), 'a') as fp:
            fp.write(f"{avg_train_loss} {avg_val_loss} {epoch}\n")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(encoder, os.path.join(model_checkpoints_folder, 'checkpoints/best_model_fine.pth'))

    print("Training complete!")
