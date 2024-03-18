import torch
import torch.nn as nn
import lightning.pytorch as pl
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModel
from transformers.modeling_utils import Conv1D
from torchmetrics import Accuracy
import math

# VAE model
class TVAE(pl.LightningModule):
    def __init__(self, config, learning_rate, rtype='CLS', max_length=256, max_epoch=5):
        super().__init__()
        self.save_hyperparameters()
        # Set parameters
        self.config = config
        self.learning_rate = learning_rate
        self.max_length=max_length
        self.max_epoch = max_epoch
        self.rtype = rtype

        # Set model

        # DistilGPT2 encoder
        self.encoder = AutoModel.from_pretrained('distilbert-base-cased')
        self.encoder.resize_token_embeddings(config.vocab_size)

        # Reduction methods and projection to posterior/prior
        if rtype == 'CLS':
            self.latent = CLSReducer(embed_dim=config.dim)
        elif rtype == 'AVG':
            self.latent = AVGReducer(embed_dim=config.dim)
        elif rtype == 'Scaling1':
            self.latent = Scaling1Reducer(embed_dim=config.dim, attention_dim=config.dim, max_length=self.max_length)
        elif rtype == 'Scaling4':
            self.latent = Scaling4Reducer(embed_dim=config.dim, attention_dim=config.dim, max_length=self.max_length)
        elif rtype == 'Pooling1':
            self.latent = Pooling1Reducer(embed_dim=config.dim, attention_dim=config.dim, max_length=self.max_length, mode='average')
        else:
            self.latent = Pooling4Reducer(embed_dim=config.dim, attention_dim=config.dim, max_length=self.max_length, mode='average')

        
        # DistilGPT2 decoder with z injected into input embedding
        # Do decoder embedding by hand to inject latent representation
        #self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        #self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        #self.drop = nn.Dropout(config.embd_pdrop)

        # latent input projection
        #self.input_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.decoder = AutoModel.from_pretrained('distilbert-base-cased')
        self.decoder.resize_token_embeddings(config.vocab_size)
        # LM head for token reconstruction
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # LM head for latent injection
        self.lm_head_z = Conv1D(config.vocab_size, config.dim)

        # Loss functional
        self.cross_entropy = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes = config.vocab_size)

    def reparameterize(self, mean, logvar, z=None):
        std = logvar.mul(0.5).exp()
        if z is None:
            z = torch.randn(std.size(), device=self.device, dtype=mean.dtype)
        return z.mul(std) + mean

    def kl_loss(self, mean1, logvar1, mean2, logvar2):
        exp = logvar1 - logvar2 - torch.pow(mean1 - mean2, 2) / logvar2.exp() - torch.exp(logvar1 - logvar2) + 1
        kl = -0.5 * torch.sum(exp, tuple(range(1, len(exp.shape))))
        return kl.mean()

    def get_beta(self, current_step, cycle_window):
        sub_cycle = cycle_window // 3
        if current_step % cycle_window == 0:
            return 0.0001

        if current_step % cycle_window < cycle_window // 3:
            return 0.0001
        elif current_step % cycle_window >= sub_cycle and current_step % cycle_window < 2* sub_cycle:
            return 0.0001 + (current_step-sub_cycle) * 0.9999 / (sub_cycle)
        else:
            return 1
        
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            ):
        # get last hidden state of encoder
        hidden_state = self.encoder(input_ids, attention_mask)[0]
        #hidden_state = hidden_state.transpose(1,0)  #batch first = false
        
        # Get latent representation
        posterior_mean, posterior_logvar = self.latent(hidden_state, attention_mask)

        # Give prior mean and logvar in form of gaussian distribution with mean 0 and std of 1 -> logvar = 0
        prior_mean = prior_logvar = torch.zeros([input_ids.size(0), self.config.dim], device=self.device)
        prior_mean, prior_logvar = prior_mean.to(posterior_mean.dtype), prior_logvar.to(posterior_logvar.dtype)

        # Get latent representation
        z = self.reparameterize(posterior_mean, posterior_logvar)
        #z = posterior_mean
        
        # Get decoder output
        # drop last token in sequence as it is the end token (target are shifted right)
        # repurposing CLS and SEP token as bos and eos tokens
        shifted_hidden_states = input_ids[:,:-1]
        if attention_mask is not None:
            attention_mask = attention_mask[:,:-1]
        
        decoder_output = self.decoder(input_ids=shifted_hidden_states, attention_mask=attention_mask)
        hidden_state = decoder_output[0]
        lm_logits = self.lm_head(hidden_state)

        # lm_logits_rep
        lm_logits_z = self.lm_head_z(z)
        if self.rtype == 'CLS' or self.rtype == 'AVG':
            lm_logits_z = lm_logits_z.unsqueeze(dim=1)
        lm_logits_z = lm_logits_z.expand(lm_logits.size())
        lm_logits = lm_logits + lm_logits_z

        # compute kl loss
        kl_loss = self.kl_loss(posterior_mean, posterior_logvar, prior_mean, prior_logvar)

        outputs = lm_logits

        return outputs, kl_loss, z, posterior_mean

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=self.max_epoch)
        return optimizer

    def training_step(self, batch, batch_idx):
        beta = self.get_beta(self.global_step, 1500)
        batch = {k: v for k, v in batch.items()}
        x = batch['input_ids']
        x_mask = batch['attention_mask']
        target_ids = batch['input_ids']
        # shift to right by dropping first token
        target_ids = target_ids[:,1:]
        
        outputs, kl_loss, z, pm = self(input_ids=x, attention_mask=x_mask)
        preds = outputs.view(-1, outputs.shape[-1])
        target_ids = target_ids.reshape(-1)
        c_loss = self.cross_entropy(preds, target_ids)
        loss = c_loss + beta * kl_loss
        acc = self.accuracy(preds, target_ids)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('c_loss', c_loss, on_step=True, on_epoch=True)
        self.log('kl_loss', kl_loss, on_step=True, on_epoch=True)
        
        self.log('train_acc', acc, on_step=True, on_epoch=True)
        return {'loss': loss, 'train_acc': acc}

    def validation_step(self, batch, batch_idx):
        beta = self.get_beta(self.global_step, 1500)
        batch = {k: v for k, v in batch.items()}
        x = batch['input_ids']
        x_mask = batch['attention_mask']
        target_ids = batch['input_ids']
        # shift to right by dropping first token
        target_ids = target_ids[:,1:]
        
        outputs, kl_loss, z, pm = self(input_ids=x, attention_mask=x_mask)
        preds = outputs.view(-1, outputs.shape[-1])
        target_ids = target_ids.reshape(-1)
        c_loss = self.cross_entropy(preds, target_ids)
        loss = c_loss + beta * kl_loss
        acc = self.accuracy(preds, target_ids)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        return {'loss': loss, 'val_acc': acc}
        
### Reduction Models
class CLSReducer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.embed_dim= embed_dim
        self.mean = Conv1D(embed_dim, embed_dim)
        self.logvar = Conv1D(embed_dim, embed_dim)

    def forward(self, hidden_states, attention_mask=None):
        # Chose first token in sequnce as CLS token
        r = hidden_states[:,0]

        mean = self.mean(r)
        logvar = self.logvar(r)
        return mean, logvar

class AVGReducer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # Average all token ins sequence of last hidden layer
        self.embed_dim= embed_dim
        self.mean = Conv1D(embed_dim, embed_dim)
        self.logvar = Conv1D(embed_dim, embed_dim)

    def forward(self, hidden_states, attention_mask=None):
        # Chose first token in sequnce as CLS token
        r = torch.mean(hidden_states, 1)

        mean = self.mean(r)
        logvar = self.logvar(r)
        return mean, logvar


class ScalingAttention(nn.Module):
    def __init__(self,
                 embed_dim,
                 attention_dim,
                 n_input_tokens,
                 n_output_tokens):
        super().__init__()
        # Reduce the length of sequence by initilazing q with trainable matrix
        self.fcq = nn.Linear(embed_dim, attention_dim, bias=False)
        self.fck = nn.Linear(embed_dim, attention_dim, bias=False)
        self.fcv = nn.Linear(embed_dim, attention_dim, bias=False)
        weights = torch.empty(n_output_tokens, n_input_tokens)
        nn.init.xavier_uniform_(weights)
        self.w_s = nn.Parameter(weights, requires_grad=True)
        self.ln = nn.LayerNorm(attention_dim)
        
    def forward(self, x):
        q = self.fcq(x)
        q = torch.matmul(self.w_s, q)
        k = self.fck(x)
        v = self.fcv(x)

        attention = F.scaled_dot_product_attention(q,k,v,dropout_p=0)
        return self.ln(attention)
    
class PoolingAttention(nn.Module):
    def __init__(self,
                 embed_dim,
                 attention_dim,
                 kernel,
                 stride,
                 mode='average',
                 dropout=0.0,
                 ):
        super().__init__()
        self.fcq = nn.Linear(embed_dim, attention_dim, bias=False)
        self.fck = nn.Linear(embed_dim, attention_dim, bias=False)
        self.fcv = nn.Linear(embed_dim, attention_dim, bias=False)
        if mode == 'average':
            self.pool = nn.AvgPool2d(kernel, stride)
        else:
            self.pool = nn.MaxPool2d(kernel, stride)
        self.ln1 = nn.LayerNorm(attention_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Get Q, K, V vectors
        q = self.fcq(x)
        k = self.fck(x)
        v = self.fcv(x)
        attn_weight = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(q.size(-1)), dim=-1)
        pooled_attn_weight = self.pool(attn_weight)
        attn = pooled_attn_weight @ v
        attn = self.dropout(attn)
        z = self.ln1(attn)
        z = self.dropout(z)
        return z

class Scaling1Reducer(nn.Module):
    def __init__(self, embed_dim, attention_dim, max_length):
        super().__init__()
        # Average all token ins sequence of last hidden layer
        self.embed_dim= embed_dim
        self.n_input = max_length
        self.red = ScalingAttention(embed_dim, attention_dim, self.n_input, 1)
        
        self.mean = Conv1D(embed_dim, embed_dim)
        self.logvar = Conv1D(embed_dim, embed_dim)

    def forward(self, hidden_states, attention_mask=None):
        # Chose first token in sequnce as CLS token
        r = self.red(hidden_states)

        mean = self.mean(r)
        logvar = self.logvar(r)
        return mean, logvar

class Pooling1Reducer(nn.Module):
    def __init__(self, embed_dim, attention_dim, max_length, mode='average'):
        super().__init__()
        # Pool attention map in a way that it reduces height to 1 and retain width
        self.embed_dim= embed_dim
        self.n_input = max_length
        self.red = PoolingAttention(embed_dim, attention_dim, kernel=[self.n_input,1], stride=[max_length,1])
        
        self.mean = Conv1D(embed_dim, embed_dim)
        self.logvar = Conv1D(embed_dim, embed_dim)

    def forward(self, hidden_states, attention_mask=None):
        # Chose first token in sequnce as CLS token
        r = self.red(hidden_states)

        mean = self.mean(r)
        logvar = self.logvar(r)
        return mean, logvar

class Scaling4Reducer(nn.Module):
    def __init__(self, embed_dim, attention_dim, max_length):
        super().__init__()
        # Average all token ins sequence of last hidden layer
        self.embed_dim= embed_dim
        self.n_input = max_length
        self.red1 = ScalingAttention(embed_dim, attention_dim, self.n_input, 3*self.n_input//4)
        self.red2 = ScalingAttention(attention_dim, attention_dim, 3*self.n_input//4, 2*self.n_input//4)
        self.red3 = ScalingAttention(attention_dim, attention_dim, 2*self.n_input//4, 1*self.n_input//4)
        self.red4 = ScalingAttention(attention_dim, attention_dim, 1*self.n_input//4, 1)
        
        self.mean = Conv1D(embed_dim, embed_dim)
        self.logvar = Conv1D(embed_dim, embed_dim)

    def forward(self, hidden_states, attention_mask=None):
        # Chose first token in sequnce as CLS token
        r = self.red1(hidden_states)
        r = self.red2(r)
        r = self.red3(r)
        r= self.red4(r)

        mean = self.mean(r)
        logvar = self.logvar(r)
        return mean, logvar

class Pooling4Reducer(nn.Module):
    def __init__(self, embed_dim, attention_dim, max_length, mode='average'):
        super().__init__()
        # Pool attention map in a way that it reduces height to 1 and retain width
        # because we fix input to 256, kernel 4 can always cleanly divide
        self.embed_dim= embed_dim
        self.n_input = max_length
        self.red1 = PoolingAttention(embed_dim, attention_dim, kernel=[4,1], stride=[4,1])
        self.red2 = PoolingAttention(attention_dim, attention_dim, kernel=[4,1], stride=[4,1])
        self.red3 = PoolingAttention(attention_dim, attention_dim, kernel=[4,1], stride=[4,1])
        self.red4 = PoolingAttention(attention_dim, attention_dim, kernel=[4,1], stride=[4,1])
        
        self.mean = Conv1D(embed_dim, embed_dim)
        self.logvar = Conv1D(embed_dim, embed_dim)

    def forward(self, hidden_states, attention_mask=None):
        # Chose first token in sequnce as CLS token
        r = self.red1(hidden_states)
        r = self.red2(r)
        r = self.red3(r)
        r = self.red4(r)

        mean = self.mean(r)
        logvar = self.logvar(r)
        return mean, logvar

class AverageSelfAttention(nn.Module):
    def __init__(self, attention_size):
        super(AverageSelfAttention, self).__init__()
        w = torch.empty(attention_size)
        nn.init.normal_(w, std=0.02)
        self.attention_weights = nn.Parameter(w)
        self.softmax = nn.Softmax(dim=-1)
        self.non_linearity = gelu

    def forward(self, inputs, attention_mask=None):

        qk = self.non_linearity(inputs.matmul(self.attention_weights))

        if attention_mask is not None:
            qk = qk + attention_mask

        qk = self.softmax(qstar)
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))
        representations = weighted.sum(1).squeeze(1)

        return representations, qk


class AvgSelfReducer(nn.Module):
    def __init__(self, attention_size):
        super().__init__()
        self.r = AverageSelfAttention(attention_size)
        
        self.mean = Conv1D(embed_dim, embed_dim)
        self.logvar = Conv1D(embed_dim, embed_dim)

    def forward(self, inputs, attention_mask=None):
        r= self.r(inputs, attention_mask)

        mean = self.mean(r)
        logvar = self.logvar(r)
        return mean, logvar
