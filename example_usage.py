import numpy as np
from tqdm import tqdm
from anti_transfer_loss import ATLoss
from torchvision import models
import torch

#parameters
at_layer = 6
aggregation_func = 'gram'
distance_func = 'cos_squared'
batch_size = 1
epochs = 10

#dummy dataset
random_predictors = torch.rand(20,3,50,50)
random_target = torch.rand(20,1000)
dataset = utils.TensorDataset(random_predictors, random_target)
data_loader = utils.DataLoader(dataset, batch_size, shuffle=True, pin_memory=True)

#create models (a pretrained model and a model to train)
model = models.vgg13()
pretrained_model = models.vgg13()
device = torch.device('cpu')
model = model.to(device)
pretrained_model = pretrained_model.to(device)
#load pre-trained checkpoint of the orthogonal model
#pretrained_path = '/Users/eric/Desktop/gram/new_repo/vgg13-c768596a.pth'
#pretrained_model = vgg.load_state_dict(torch.load(pretrained_path), strict=False)

#init optimizer and main loss
criterion = nn.MSELoss()
optimizer = optim.Adam(params=model.parameters())

#init AT loss with the pre-trained model
AT = ATLoss(pretrained_model.features)

#training loop
for i in range(epochs):
    print("Epoch " + str(i))
    model.train()
    for example_num, (x, target) in enumerate(data_loader):
        target = target.to(device)
        x = x.to(device)
        optimizer.zero_grad()
        outputs = model(x)

        #compute main loss
        loss = criterion(outputs, target)

        #compute AT loss with the desired parameters
        at_loss = AT.loss(x,                      #input batch
                          model.features,       #current model
                          at_layer=18,            #leyer to compute AT loss
                          beta=1.,                #weight parameter
                          aggregation='gram',     #channel aggregation
                          distance='cos_squared') #distance function

        total_loss = loss + at_loss
        total_loss.backward()
        optimizer.step()
        print("Total loss: ",  total_loss.item(), '| AT loss: :' , at_loss)
        '''
        With randomly-initialized models the AT loss is low from the beginning, as
        features are uncorrelated. AT learning helps to prevent to develop such correlation.
        '''
