import random
import torch
import numpy as np
from arguments import parse_arguments 
import os
from copy import deepcopy
import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter
# import sys
# sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data'))
# get arguments
args = parse_arguments()

# Set hyperparams:
lr = args.lr
missing = False
num_epochs = args.num_epochs
batch_size = args.batch_size
anneal_factor = args.anneal_factor
patience = args.patience
base_path = os.path.join(args.log_dir, args.model_type)

seeds = [0,42,1346,325,1243,76,423,567,34,534,46,456,346,12,239]
# seeds = [0,42]
macro_f1_all_0 = []
weighted_f1_all_0 = []
macro_precision_all_0 = []
macro_recall_all_0 = []

macro_f1_all_2 = []
weighted_f1_all_2 = []
macro_precision_all_2 = []
macro_recall_all_2 = []

for seed in seeds:

    # set random seeds
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Other imports now
    from utils import *
    from Attention import *
    from model import *
    from data_utils import *
    from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

    TRAIN = args.train
    TEST = args.test
    VAL_SPLIT_PATH = args.val_split_path

    # Load datasets
    train_dataset = MICDataset(TRAIN)
    train_val_split = np.load(args.val_split_path)
    train_data = Subset(train_dataset, train_val_split['train'])
    val_data = Subset(train_dataset, train_val_split['val'])
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size)

    test_dataset = MICDataset(TEST)
    test_loader = DataLoader(test_dataset, batch_size)


    #-------------------#
    # Model Definition  #
    #-------------------#
    if args.model_type == 'attention':
        model = DecisionLevelSingleAttention(
                freq_bins=128,
                classes_num=20,
                emb_layers=args.emb_layers,
                hidden_units=args.hidden_size,
                drop_rate=args.dropout_rate)
    elif args.model_type == 'FC':
        model = FC()
    elif args.model_type == 'FC_T':
        model = FC_T()
    elif args.model_type == 'RNN':
        model = BaselineRNN_2()
    model = model.cuda()

    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=args.wd)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer, 
    #         factor=anneal_factor,
    #         patience=patience,
    #         threshold=1e-3)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30], gamma = 0.1)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
            # factor=anneal_factor, patience=patience, threshold=0.05)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, 1e-5)
    # criterion = torch.nn.BCEWithLogitsLoss(reduction='none').cuda()

    # if args.loss_type == 'BCE':
    #     criterion = BCE()
    # elif args.loss_type == 'PartialBCE':
    #     criterion = PartialBCE(args.alpha, args.beta, args.gamma)
    criterion = torch.nn.BCELoss()

    # # Make logs
    # log_dir = '../../log/ISMIR_missing/Discriminative/'
    # # time_string = datetime.datetime.now().isoformat()
    # time_string = ""
    # run_name = "{}_{}_missing_{}_lr_{}_lr_decay_{}_patience_{}\
        # ".format(args.id, model.__class__.__name__, missing, lr, 
        # anneal_factor, patience)

    # remove_dir(os.path.join(log_dir, run_name))
    # writer = SummaryWriter(os.path.join(log_dir, run_name))
    writer_path = os.path.join(base_path, args.id+'_seed_'+str(seed))
    remove_dir(writer_path)
    print("Dumping logs in {}".format(writer_path))
    writer = SummaryWriter(writer_path)

    best_models = [None, None, None]
    best_val_loss = 100000.0
    best_f1_score_w = -1.0
    best_f1_score_m = -1.0
    try:
        for epoch in tqdm(range(num_epochs)):
            # Train model
            loss = discriminative_trainer(
                    model=model,
                    data_loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion)
            # log in tensorboard
            writer.add_scalar('Training/Prediction Loss', loss, epoch)
            
            # Eval model
            loss, avg_f1_weighted, avg_f1_macro,_,_,_ = discriminative_evaluate(model, val_loader, criterion)
            # log in tensorboard
            writer.add_scalar('Validation/Prediction Loss', loss, epoch)
            writer.add_scalar('Validation/Average F1-score Weighted', avg_f1_weighted.mean(), epoch)
            writer.add_scalar('Validation/Average F1-score Macro', avg_f1_macro.mean(), epoch)

            #Save best models
            if loss < best_val_loss:
                best_val_loss = loss
                best_models[0] = deepcopy(model)
            
            if avg_f1_weighted.mean() > best_f1_score_w:
                best_f1_score_w = avg_f1_weighted.mean()
                best_models[1] = deepcopy(model)
            
            if avg_f1_macro.mean() > best_f1_score_m:
                best_f1_score_m = avg_f1_macro.mean()
                best_models[2] = deepcopy(model)

            # Anneal LR
            # scheduler.step()
    except KeyboardInterrupt:
        print('Stopping training. Now testing')

    # Test the model
    for i, model in enumerate(best_models):
        loss, avg_f1_weighted, avg_f1_macro, predictions, avg_p_macro, avg_r_macro = discriminative_evaluate(model, test_loader, criterion)
        print('Test Prediction Loss: ', loss.item())
        print('Test Avg F1-score weighted: ', avg_f1_weighted.mean())
        print('Test Avg F1-score macro: ', avg_f1_macro.mean())
        print('Test Class F1-score weighted: ', avg_f1_weighted)
        print('Test Class F1-score macro: ', avg_f1_macro)
        scores_weighted = np.append(avg_f1_weighted, avg_f1_weighted.mean())
        scores_macro = np.append(avg_f1_macro, avg_f1_macro.mean())
        macro_precision = np.append(avg_p_macro, avg_p_macro.mean())
        macro_recall = np.append(avg_r_macro, avg_r_macro.mean())
        scores = np.concatenate((scores_weighted.reshape(-1, 1), scores_macro.reshape(-1, 1)), axis=1)
        if i == 0:
            writer.add_scalar('Testing/F1-score Macro Best Val Loss', avg_f1_macro.mean(), 0)
            with open(os.path.join(writer_path, 'best_val_loss.txt'), 'w') as f:
                torch.save(model.state_dict(), os.path.join(writer_path, 'best_val_loss.pth'))
                np.savetxt(f, scores)
                np.save(os.path.join(writer_path, 'best_val_loss.preds'), predictions)
                macro_f1_all_0.append(scores_macro)
                weighted_f1_all_0.append(scores_weighted)
                macro_precision_all_0.append(macro_precision)
                macro_recall_all_0.append(macro_recall)
        elif i == 1:
            writer.add_scalar('Testing/F1-score Macro Best F1 weighted', avg_f1_macro.mean(), 0)
            with open(os.path.join(writer_path, 'best_f1_score_weighted.txt'), 'w') as f:
                torch.save(model.state_dict(), os.path.join(writer_path, 'best_f1_score_weighted.pth'))
                np.savetxt(f, scores)
                np.save(os.path.join(writer_path, 'best_f1_score_weighted.preds'), predictions)
        elif i == 2:
            writer.add_scalar('Testing/F1-score Macro Best F1 macro', avg_f1_macro.mean(), 0)
            with open(os.path.join(writer_path, 'best_f1_score_macro.txt'), 'w') as f:
                torch.save(model.state_dict(), os.path.join(writer_path, 'best_f1_score_macro.pth'))
                np.savetxt(f, scores)
                np.save(os.path.join(writer_path, 'best_f1_score_macro.preds'), predictions)
                macro_f1_all_2.append(scores_macro)
                weighted_f1_all_2.append(scores_weighted)
                macro_precision_all_2.append(macro_precision)
                macro_recall_all_2.append(macro_recall)
                
                # For this one, also save the predictions for the full training set.
                # train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
                # train_predictions = model_forward(model, train_loader)
                # torch.save(train_predictions, os.path.join(writer_path, 'predictions_best_f1_score_macro.pth'))

macro_f1_all_0 = np.array(macro_f1_all_0)
weighted_f1_all_0 = np.array(weighted_f1_all_0)
macro_precision_all_0 = np.array(macro_precision_all_0)
macro_recall_all_0 = np.array(macro_recall_all_0)
macro_f1_all_2 = np.array(macro_f1_all_2)
weighted_f1_all_2 = np.array(weighted_f1_all_2)
macro_precision_all_2 = np.array(macro_precision_all_2)
macro_recall_all_2 = np.array(macro_recall_all_2)

np.savetxt(os.path.join(base_path, 'macro_f1_all_0.txt'), macro_f1_all_0.T)
np.savetxt(os.path.join(base_path, 'weighted_f1_all_0.txt'), weighted_f1_all_0.T)
np.savetxt(os.path.join(base_path, 'macro_precision_all_0.txt'), macro_precision_all_0.T)
np.savetxt(os.path.join(base_path, 'macro_recall_all_0.txt'), macro_recall_all_0.T)

np.savetxt(os.path.join(base_path, 'macro_f1_all_2.txt'), macro_f1_all_2.T)
np.savetxt(os.path.join(base_path, 'weighted_f1_all_2.txt'), weighted_f1_all_2.T)
np.savetxt(os.path.join(base_path, 'macro_precision_all_2.txt'), macro_precision_all_2.T)
np.savetxt(os.path.join(base_path, 'macro_recall_all_2.txt'), macro_recall_all_2.T)