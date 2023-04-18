import copy
import torch
import torch.nn as nn

class WrapLayer(nn.Module):
  def __init__(self, layer, max_seq_len):
    super().__init__()
    self.layer = layer
    self.position_ids = nn.Parameter(torch.arange(max_seq_len).unsqueeze(0).view(-1, max_seq_len), requires_grad=False)

  def forward(self, x):
    _, seq_len, h = x.shape
    return self.layer(x, position_ids=self.position_ids[:, :seq_len])[0]


class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)

class AdapterModel(nn.Module):
  def __init__(self, pretrained_model, max_seq_len, adapter_size, adapter_dropout=0.1, init_with_existing_tokens=False): # add cast to 8bit param
    super().__init__()
    # Copy token embedding layer from the pretrained model
    vocab_size = pretrained_model.config.vocab_size
    hidden_size = pretrained_model.config.hidden_size
    self.token_emb = nn.Embedding(vocab_size, hidden_size)
    self.token_emb.weight = copy.deepcopy(pretrained_model.gpt_neox.embed_in.weight)
    self.token_emb.weight.requires_grad = False
    pretrained_model.gpt_neox.embed_in = None
    
    # Initialize adapter from normal distribution
    self.adapter = nn.Parameter(torch.randn(adapter_size, hidden_size) * 0.002, requires_grad=True)
    self.adapter_dropout = nn.Dropout(adapter_dropout)

    # Freeze pretrained layers
    for param in pretrained_model.parameters():
      param.requires_grad = False
      if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)

    # Make transformer layers sequential
    self.transformer_layers = nn.Sequential(*[
        WrapLayer(layer, max_seq_len) for layer in pretrained_model.gpt_neox.layers
    ])

    # Output layers -- cast final output to fp32
    self.out = nn.Sequential(
        pretrained_model.gpt_neox.final_layer_norm,
        CastOutputToFloat(pretrained_model.embed_out)
    )
    
  def forward(self, x):
    bsz, _ = x.shape
    token_emb = self.token_emb(x) # bsz, inp_len, embed_dim
    adapter_emb = self.adapter_dropout(self.adapter.unsqueeze(0).repeat(bsz, 1, 1)) # bsz, adapter_size, embed_dim
    seq = torch.cat([adapter_emb, token_emb], dim=1)
    seq =  torch.utils.checkpoint.checkpoint_sequential(self.transformer_layers, 3, seq)
    return self.out(seq)
    
