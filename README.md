# AttentionMIC

This repo consists of the code accompanying the ISMIR 2019 paper: 

Siddharth Gururani, Mohit Sharma, Alexander Lerch. An Attention Mechanism for Musical Instrument Recognition. (To appear) In Proceedings of the International Society of Music Information Retrieval, ISMIR 2019.

# Data

Before you run any code, please download the data from [here](https://drive.google.com/open?id=1feFBcMAe80Qy_EAYhamxNTfW46ICfEOn
). You will then place *train.npz* and *test.npz* in the data folder.

Alternatively, you may download the OpenMIC dataset and use the tool `data/data_split.py` to generate the dataset splits.

# Prerequities

You need to have `Pytorch, TensorboardX, Tqdm, Deepcopy` installed in your python environment. I will update the repo with a conda environment file for easy setup.
By default the code assumes the presence of a GPU. I will add a device-agnostic version of the code in future commits.

# Usage

The commands in the `multirun_commands.txt` file were used to train the different models with various random seeds. If you are only interested in the attention model, that resides in `Attention.py`. The baseline models are implemented in `model.py`.

# Acknowledgement

We thank Qiuqiang Kong for their [implementation](https://github.com/qiuqiangkong/audioset_classification) of the attention model from their paper:

Qiuqiang Kong, Yong Xu, Wenwu Wang and Mark D. Plumbley. Audio Set classification with attention model: A probabilistic perspective. In: International Conference on Acoustics, Speech, and Signal Processing, ICASSP 2018, Calgary, Canada, 15-20 April 2018.
