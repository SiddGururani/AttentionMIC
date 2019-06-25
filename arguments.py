import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--id', default='main',
        help='Identifier used for tensorboard visualization.')
parser.add_argument('--seed', type=int, default=0,
        help='Random seed')

# Training related hyperparameters
parser.add_argument('--batch_size', type=int, default=128,
        help='Batch size.')
parser.add_argument('--num_epochs', type=int, default=250,
        help='Number of epochs')
parser.add_argument('--loss_type', default='BCE',
        help='Whether to use BCE or PartialBCE')
parser.add_argument('--alpha', type=float, default=1.,
        help='alpha parameter for PartialBCE')
parser.add_argument('--beta', type=float, default=0.,
        help='beta parameter for PartialBCE')
parser.add_argument('--gamma', type=float, default=-1.,
        help='gamma parameter for PartialBCE')


# Optimizer related hyper parameter
parser.add_argument('--anneal_factor', type=float, default=0.5,
        help='Annealing Factor used for learning rate scheduler.')
parser.add_argument('--patience', type=int, default=3,
        help='Pateince used for learning rate scheduler.')
parser.add_argument('--lr', type=float, default=0.0005,
        help='Learning rate for the optimizer.')
parser.add_argument('--wd', type=float, default=1e-5,
        help="Weight decay for the network parameters")

# Model related hyperparameters
parser.add_argument('--model_type', default='attention',
        help='Which type of model to train')
parser.add_argument('--emb_layers', type=int, default=3,
        help='Number of embedding layers')
parser.add_argument('--hidden_size', type=int, default=128,
        help='Hidden size for embedding layers')
parser.add_argument('--dropout_rate', type=float, default=0.5,
        help='probability of zeroing out a hidden node')

# Data related hyperparameters
parser.add_argument('--train', default='./data/train.npz',
        help='train split for the dataset.')
parser.add_argument('--test', default='./data/test.npz',
        help='test split for the dataset.')
parser.add_argument('--val_split_path', default='./data/train_val.split',
        help='train and validation indices for the training split.')      
parser.add_argument('--log_dir', default='log',
        help='Directory for tensorboard logs')

def parse_arguments():
    return parser.parse_args()
    