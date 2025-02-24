import torch
from tqdm import tqdm
torch.manual_seed(3407)

def get_features_from_encoder(encoder, loader):
    x_train = []
    y_train = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # get the features from the pre-trained model
    for i, (x, y) in enumerate(tqdm(loader)):
        x = x.float()
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            feature_vector = encoder(x)
            #print(feature_vector)
            x_train.extend(feature_vector)
            y = y.cpu()
            y_train.extend(y.numpy())

    x_train = torch.stack(x_train)
    y_train = torch.tensor(y_train)
    return x_train, y_train
