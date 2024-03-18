import torch
import wandb
import os
import lightning.pytorch as pl
from models.TVAE import TVAE
from data.ArxivDataModule import ArxivDataModule
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from transformers import AutoConfig
import numpy as np
import random

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

modelpath = 'trained_models/'
models = ['CLS-v1', 'AVG-v1', 'Scaling1-v1', 'Scaling4-v1', 'Pooling1-v1', 'Pooling4-v1']

data = ArxivDataModule(32, 256)
data.prepare_data()
loader = data.train_dataloader()

random.seed(0)
embs = np.array([])
labels = []
for m in models:
    model = TVAE.load_from_checkpoint(modelpath + m +'.ckpt').to('cuda')
    model.eval()

    for i, batch in enumerate(loader):
        if i >= 400:
            break
        batch = {k: v for k, v in batch.items()}
        data = batch['input_ids'].cuda()
        lab = batch['category'].cpu().detach()
        mask = batch['attention_mask'].cuda()

        out, kl, z, pm = model(data, mask)

        pm = pm.squeeze().cpu().detach()
        if i == 0:
            embs = pm
            labels = lab
        else:
            embs = torch.cat((embs, pm), dim=0)
            labels = torch.cat((labels,lab),dim=0)

    embs = embs.numpy()
    labels = labels.numpy()
    print(labels.shape)
    print(embs.shape)
    print('Start tsne'+m)
    z_emb_2d = TSNE(n_components=2, perplexity=40, n_iter=1000, verbose=1).fit_transform(embs)

    # Remove outliers perhaps
    def remove_outliers(data, r=2.0):
        outliers_data = abs(data - np.mean(data, axis=0)) >= r * np.std(data, axis=0)
        outliers = np.any(outliers_data, axis=1)
        keep = np.logical_not(outliers)
        return outliers, keep

    outliers, keep = remove_outliers(z_emb_2d)
    z_emb_2d = z_emb_2d[keep, :]
    y = [l for l, k in zip(labels, keep.tolist()) if k]

    # plot
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_axes([0, 0, 1, 1])
    cc = ['r', 'b', 'g']
    for i, l in enumerate(sorted(set(y))):
        idx = [yl == l for yl in y]
        meanx, meany = np.mean(z_emb_2d[idx,0]), np.mean(z_emb_2d[idx,1])
        stdx, stdy = np.std(z_emb_2d[idx,0]), np.std(z_emb_2d[idx,1])
        print(meanx,meany)
        print(stdx,stdy)
        plt.scatter(z_emb_2d[idx,0], z_emb_2d[idx,1], c=cc[i], s=10., edgecolor='none', alpha=0.3)
        plt.scatter(meanx, meany, c=cc[i], s=50., edgecolor='black', alpha=1.)
    plt.savefig(os.path.join('figures/', m + '.pdf'))
    plt.close(fig)
