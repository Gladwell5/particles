import pandas as pd
import torch
from torch.utils.data import Dataset

class ParticleDS(Dataset):
    def __init__(self, csv_file, predictors, outcome):
        df = pd.read_csv(csv_file, low_memory=False)
        self.X = torch.tensor(df[predictors].values, dtype=torch.float32)
        sorted_y_values = sorted(list(set(df[outcome])))
        self.y = [sorted_y_values.index(value) for value in df[outcome].values]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Unlike traditional models, which only have a training loop that stops when the models 'converge'...
# ...Neural nets are very flexible so they can adapt too much to the training data...
# ...so we use both a training loop to train and a test loop to check when we need to stop the training

def train_loop(dataloader, model, loss_fn, optimiser, batch_size):
    size = len(dataloader.dataset)
    model.train() # set model to training mode
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X) # get model predictions
        loss = loss_fn(pred, y) # compare predictions to actual
        loss.backward() # calculate adjustments to the model parameters
        optimiser.step() # apply the adjustments
        optimiser.zero_grad() # zero out changes (so they don't accumulate)

        # print updates every 100 batches
        if batch % int(size/(5 * batch_size)) == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # set model to evaluation mode
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad(): # ensure no changes occur to the model during this loop
        for X, y in dataloader:
            pred = model(X) # get predictions
            test_loss += loss_fn(pred, y).item() # add to total loss over test set
            correct += (pred.argmax(1) == y).type(torch.float32).sum().item() # add to total correct over test set

    # calculate and show average loss and % correct
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")