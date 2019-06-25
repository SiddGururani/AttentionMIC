import os
import errno
import shutil
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import classification_report

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def to_numpy(x):
    return x.detach().cpu().numpy()

def get_class_map(rev=False):
    class_map = {'accordion': 0, 'banjo': 1, 'bass': 2, 'cello': 3, 'clarinet': 4, 'cymbals': 5,
                'drums': 6, 'flute': 7, 'guitar': 8, 'mallet_percussion': 9, 'mandolin': 10,
                'organ': 11, 'piano': 12, 'saxophone': 13, 'synthesizer': 14, 'trombone': 15,
                'trumpet': 16, 'ukulele': 17, 'violin': 18, 'voice': 19}
    rev_class_map = {v: k for k, v in class_map.items()}
    return class_map if not rev else rev_class_map
    
def make_dir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def remove_dir(path):
    # check if folder exists
    if os.path.exists(path):
        # remove if exists
        print('Deleting {}'.format(path))
        shutil.rmtree(path)
    
def cuda(X):
    if type(X) is list:
        X = X[0].cuda(), X[1].cuda()
    else: 
        X = X.cuda()
    return X

def discriminative_trainer(model, data_loader, optimizer, criterion, inst=None):
    model.train()
    loss_tracker = AverageMeter()
    for (X, _, Y_true, Y_mask) in tqdm(data_loader):
        X = cuda(X)
        Y_true = Y_true.cuda()
        Y_mask = Y_mask.cuda()
        if inst is not None:
            Y_true = Y_true[:,inst].view(-1,1)
            Y_mask = Y_mask[:,inst].view(-1,1)
        outputs = model(X)
        if inst is None:
            loss = criterion(outputs[Y_mask], Y_true[Y_mask])
            # loss = criterion(outputs, Y_true, Y_mask, None)
        else:
            loss = criterion(outputs[Y_mask], Y_true[Y_mask])
            loss = loss.mean()
        # loss = loss[Y_mask].mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update average meter
        loss_tracker.update(loss.item())
    return loss_tracker.avg
    
def model_forward(model, data_loader, inst=None):
    model.eval()
    if inst is not None:
        n_inst = 1
    else:
        n_inst = 20
    all_predictions = torch.Tensor(0, n_inst)
    for (X, _, _, _) in tqdm(data_loader):
        X = X.cuda()
        outputs = model(X)
        all_predictions = torch.cat((all_predictions, outputs.detach().cpu()))
    return torch.sigmoid(all_predictions)

def discriminative_evaluate(model, data_loader, criterion, inst=None):
    model.eval()
    loss_tracker = AverageMeter()
    if inst is not None:
        n_inst = 1
    else:
        n_inst = 20
    all_y_true = torch.Tensor(0, n_inst)
    all_y_mask = torch.ByteTensor(0, n_inst)
    all_predictions = torch.Tensor(0, n_inst)
    for (X, _, Y_true, Y_mask) in tqdm(data_loader):
        X = cuda(X)
        Y_true = Y_true.cuda()
        Y_mask = Y_mask.cuda()
        if inst is not None:
            Y_true = Y_true[:,inst].view(-1,1)
            Y_mask = Y_mask[:,inst].view(-1,1)
        outputs = model(X)
        if inst is None:
            loss = criterion(outputs[Y_mask], Y_true[Y_mask])
            # loss = criterion(outputs, Y_true, Y_mask)
        else:
            loss = criterion(outputs[Y_mask], Y_true[Y_mask])
            # loss = loss.mean()
        # loss = loss[Y_mask].mean()
        # Store the outputs and target for classification metric computation
        all_y_true = torch.cat((all_y_true, Y_true.detach().cpu()))
        all_y_mask = torch.cat((all_y_mask, Y_mask.detach().cpu()))
        all_predictions = torch.cat((all_predictions, outputs.detach().cpu()))
        # Update average meter
        loss_tracker.update(loss.item())
    # Get classwise classification metrics
    avg_fscore_weighted = []
    avg_fscore_macro = []
    avg_precision_macro = []
    avg_recall_macro = []
    # First convert aggregated tensors to numpy arrays for use in scikit-learn later
    all_y_mask = to_numpy(all_y_mask)
    all_y_true = to_numpy(all_y_true)
    # all_predictions = to_numpy(torch.sigmoid(all_predictions))
    all_predictions = to_numpy(all_predictions)
    
    if inst is None:
        for i in range(20):
            results = compute_accuracy_metrics(all_y_true[:,i], all_y_mask[:,i], all_predictions[:,i])
            avg_fscore_weighted.append(results['weighted avg']['f1-score'])
            avg_fscore_macro.append(results['macro avg']['f1-score'])
            avg_precision_macro.append(results['macro avg']['precision'])
            avg_recall_macro.append(results['macro avg']['recall'])
    else:
        results = compute_accuracy_metrics(all_y_true, all_y_mask, all_predictions)
        avg_fscore_weighted.append(results['weighted avg']['f1-score'])
        avg_fscore_macro.append(results['macro avg']['f1-score'])
        avg_precision_macro.append(results['macro avg']['precision'])
        avg_recall_macro.append(results['macro avg']['recall'])
    return loss, np.array(avg_fscore_weighted), np.array(avg_fscore_macro), all_predictions, np.array(avg_precision_macro), np.array(avg_recall_macro)

def compute_accuracy_metrics(labels, labels_mask, predictions, threshold=0.5):
    # if threshold is None, then find the best threshold for this data. 
    # Normally, I would get the best threshold from validation and apply to testing
    
    # Get relevant indices from the mask
    relevant_inds = np.where(labels_mask)[0]
    
    # Binarize the predictions based on the threshold.
    predictions[predictions >= threshold] = 1
    predictions[predictions < 1] = 0
    print(classification_report(labels[relevant_inds], predictions[relevant_inds]))
    # return classification report
    return classification_report(labels[relevant_inds], predictions[relevant_inds], output_dict=True)

# From https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:

# MIT License

# Copyright (c) 2018 Bjarte Mehus Sunde

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score or np.abs(score - self.best_score) < 1e-4:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


# Not using this
class BCE:
    def __init__(self):
        pass
    def __call__(self, Y_hat, Y_true, Y_mask, weights=None):
        criterion = torch.nn.BCELoss(reduction='none').cuda()
        loss = criterion(Y_hat, Y_true)
        if weights is not None:
            loss = loss*weights
        loss = loss*Y_mask.float()
        loss = loss.mean()
        return loss


# Not using this either since I am not setting different hyperparams 
# compared to using standard BCE only computed for available labels
class PartialBCE:
    def __init__(self, alpha, beta, gamma):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    def __call__(self, Y_hat, Y_true, Y_mask, weights=None):
        N, C = Y_hat.size()
        criterion = torch.nn.BCELoss(reduction='none').cuda()
        loss = criterion(Y_hat, Y_true)
        if weights is not None:
            loss = loss*weights
        loss = loss*Y_mask.float()
        known_ys = Y_mask.float().sum(1)
        p_y = known_ys/C
        g_p_y = self.alpha*(p_y**self.gamma) + self.beta
        loss = ((g_p_y/C)*loss.sum(1)).mean()
        return loss
