import os, sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import HierarchialAttentionNetwork
from dataset import News20Dataset
from dataloader import MyDataLoader
from trainer import Trainer


def train(config, device):
    dataset = News20Dataset(config.cache_data_dir, config.vocab_path, is_train=True)
    dataloader = MyDataLoader(dataset, config.batch_size)

    model = HierarchialAttentionNetwork(
        num_classes=dataset.num_classes,
        vocab_size=dataset.vocab_size,
        embed_dim=config.embed_dim,
        word_gru_hidden_dim=config.word_gru_hidden_dim,
        sent_gru_hidden_dim=config.sent_gru_hidden_dim,
        word_gru_num_layers=config.word_gru_num_layers,
        sent_gru_num_layers=config.sent_gru_num_layers,
        word_att_dim=config.word_att_dim,
        sent_att_dim=config.sent_att_dim).to(device)

    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)

    criterion = nn.NLLLoss(reduction='sum').to(device)

    trainer = Trainer(config, model, optimizer, criterion, dataloader)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bug squash for Hierarchical Attention Networks')

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--max_grad_norm", type=float, default=5)

    parser.add_argument("--embed_dim", type=int, default=100)
    parser.add_argument("--word_gru_hidden_dim", type=int, default=100)
    parser.add_argument("--sent_gru_hidden_dim", type=int, default=100)
    parser.add_argument("--word_gru_num_layers", type=int, default=1)
    parser.add_argument("--sent_gru_num_layers", type=int, default=1)
    parser.add_argument("--word_att_dim", type=int, default=200)
    parser.add_argument("--sent_att_dim", type=int, default=200)
    
    parser.add_argument("--vocab_path", type=str, default="data/glove/glove.6B.100d.txt")
    parser.add_argument("--cache_data_dir", type=str, default="data/news20/")

    config = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Make necessary data directories at the very first run
    if not os.path.exists(os.path.dirname(config.vocab_path)):
        for dir in [os.path.dirname(config.vocab_path), config.cache_data_dir]:
            os.makedirs(dir, exist_ok=True)
        print("Finished making data directories.")
        print("Before proceeding, please put the GloVe text file under data/glove as instructed.")
        print("Ending this run.")
        sys.exit()

    train(config, device)