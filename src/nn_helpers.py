from collections import Counter
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

def train_loop(dataloader, model, loss_fn, optimiser, batch_size, print_freq=3):
    model.train() # set model to training mode
    size = len(dataloader.dataset)
    if print_freq > 0:
        update_denominator = int(size/(print_freq * batch_size))
    else:
        update_denominator = 0
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X) # get model predictions
        loss = loss_fn(pred, y) # compare predictions to actual
        loss.backward() # calculate adjustments to the model parameters
        optimiser.step() # apply the adjustments
        optimiser.zero_grad() # zero out the changes (so they don't accumulate)

        # print updates print_freq times in the loop
        if update_denominator > 0 and batch > batch_size and batch % update_denominator == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return loss

def test_loop(dataloader, model, loss_fn, loud=False):
    model.eval() # set model to evaluation mode
    size = len(dataloader.dataset)
    y_counts = dict(Counter(dataloader.dataset.y))
    y_correct = {k:0 for k in y_counts.keys()}
    
    n_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad(): # ensure no changes occur to the model during this loop
        for X, y in dataloader:
            pred = model(X) # get predictions
            test_loss += loss_fn(pred, y).item() # add to total loss over test set
            hits = (pred.argmax(1) == y)
            batch_y_counts = Counter([_y.item() for hit, _y in zip(hits, y) if hit])
            for k, v in batch_y_counts.items():
                y_correct[k] += v
            correct += hits.type(torch.float32).sum().item() # add to total correct over test set

    # calculate and show average loss and % correct
    avg_acc = sum([y_correct[k]/n for k, n in y_counts.items()])/len(y_counts)
    test_loss /= n_batches
    correct /= size
    if loud:
        print(f"Test accuracy: {(100 * correct):>0.1f}%, weighted accuracy: {(100 * avg_acc):>0.1f}%, avg loss: {test_loss:>6f}")
    return correct, test_loss, avg_acc