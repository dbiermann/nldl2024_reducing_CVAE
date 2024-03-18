import torch
import wandb
import lightning.pytorch as pl
from models.TVAE import TVAE
from data.ArxivDataModule import ArxivDataModule
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import AutoConfig

def train(config=None):
    wandb.init(config=config, project=config['project_name'])
    config = wandb.config
    sim_name = 'TVAE ' + config['rtype']
    wandb.run.name = sim_name
    # Set seed
    pl.seed_everything(config['seed'], workers=True)
    
    # Load dataset
    arxivset = ArxivDataModule(config['batch_size'], config['max_length'],)
    vocab_size = arxivset.get_vocab_size()
    
    # Define model
    distil_config = AutoConfig.from_pretrained('distilbert-base-cased')
    distil_config.vocab_size = vocab_size
    print(distil_config)
                
    model = TVAE(
        distil_config,
        config['learning_rate'],
        config['rtype'],
        config['max_length'],
        config['max_epochs'])

    # Define loggers
    early_stopping = EarlyStopping(monitor='val_acc', mode='max', patience=config['patience'])
    checkpoint = ModelCheckpoint(monitor='val_acc',
                                 dirpath = 'trained_models/',
                                 filename = config['rtype'])
    wandb_logger = WandbLogger()
    
    # Create trainer
    wandb_logger.watch(model)
    trainer = pl.Trainer(devices=1, accelerator='gpu', callbacks=[early_stopping, checkpoint], logger=wandb_logger, max_epochs=config['max_epochs'], gradient_clip_val=1, gradient_clip_algorithm='norm')
    wandb.define_metric('val_acc', summary='max')
    # Train model
    trainer.fit(model, arxivset)
    
    wandb_logger.experiment.unwatch(model)
    wandb.finish()

default_config = {
    'learning_rate': 0.0001,
    'max_length': 256,
    'batch_size': 32,
    'patience': 5,
    'max_epochs': 5,
    'rtype': 'Pooling4',   # 'AVG', 'Scaling1', 'Scaling4', 'Pooling1', 'Pooling4'],
    'project_name': 'NLDL_project',
    'seed': 10
    }

if __name__ == '__main__':
        train(default_config)
