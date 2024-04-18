import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from config import get_default_config

class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 # each attention head attents to n_embd / n_head elements. We get this by multipling the n_embd matrix by a Q,K,V matrix of size n_embd, n_embd / n_head.
        # each of these Q, K, V matrices is specific to the attention head, so in essence each is attending to the same thing
        # at the end we concatenate everything back together

        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3) # this is the Q, K, V matrix, in a batch
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) # this is the output matrix
    
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        assert config.n_embd % config.n_head == 0, f"Embedding dimensions {config.n_embd} must be divisible by the number of heads {config.n_head}"

        # drop out neurons for modularity. can set to 0 if we don't like
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):

        B, T, C = x.size() # B is batch size, T is sequence length, C is embedding dimensions 
        assert C == self.n_embd, f"Input embedding dimensions {C} does not match the model embedding dimensions {self.n_embd}"
        # B sentences, each sentence has T words, each word is represented in C dimensions
        x_proj = self.c_attn(x) # This will make up the Q, K, V matrix. Sends x to 3x dimensions for each matri
        q, k, v = x_proj[ : , : , :C ], x_proj[ : , : , C:2*C ], x_proj[ : , : , 2*C:3*C ] # split the 3x dimensions into Q, K, V. all same size as x
        
        n_head = self.n_head
        # split up into n_head attention heads. each will be applied on a different part of the input
        q = q.reshape(B, n_head, T, int(C/n_head)) # (B, nh, T, hs)
        k = k.reshape(B, n_head, T, int(C/n_head))
        v = v.reshape(B, n_head, T, int(C/n_head))

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T). doing sig(q_i*k_j)/sqrt(len(k)) to find s_ij (NLP notes)

        mask = torch.ones(T,T)*float('-inf')
        mask = torch.triu(mask, diagonal=1) #can't look ahead. in first word can only look at first word. not ahead
        mask = mask.repeat(B, n_head, 1, 1)
        att = att + mask
        att = F.softmax(att, dim=-1)


        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs). z_i = sum a_ij v_j
        y = y.transpose(1, 2).contiguous().view(B, T, C) # concatenate the n_head attention heads
        y = self.c_proj(y)
        y = self.resid_dropout(y)

        return y
    
    def get_attn(self, x):
        B, T, C = x.size() # B is batch size, T is sequence length, C is embedding dimensions 
        assert C == self.n_embd, f"Input embedding dimensions {C} does not match the model embedding dimensions {self.n_embd}"
        # B sentences, each sentence has T words, each word is represented in C dimensions
        x_proj = self.c_attn(x) # This will make up the Q, K, V matrix. Sends x to 3x dimensions for each matri
        q, k, v = x_proj[ : , : , :C ], x_proj[ : , : , C:2*C ], x_proj[ : , : , 2*C:3*C ] # split the 3x dimensions into Q, K, V. all same size as x
        
        n_head = self.n_head
        # split up into n_head attention heads. each will be applied on a different part of the input
        q = q.reshape(B, n_head, T, int(C/n_head)) # (B, nh, T, hs)
        k = k.reshape(B, n_head, T, int(C/n_head))
        v = v.reshape(B, n_head, T, int(C/n_head))

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T). doing sig(q_i*k_j)/sqrt(len(k)) to find s_ij (NLP notes)

        mask = torch.ones(T,T)*float('-inf')
        mask = torch.triu(mask, diagonal=1) #can't look ahead. in first word can only look at first word. not ahead
        mask = mask.repeat(B, n_head, 1, 1)
        att = att + mask
        att = F.softmax(att, dim=-1)
        return att
    
    
class Block(nn.Module):
    # a block is a transformer block. It has a self attention layer, a feed forward layer, and a layer norm
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc = nn.Linear(config.n_embd, config.n_embd * 4),
            c_proj = nn.Linear(config.n_embd * 4, config.n_embd),
            act = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop)
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # feed forward layer. just fancy shmancy syntax, very basic tho


    def forward(self, x, layernorm = False, return_hs = False, insert = {}):
        #BIMT doesnt use layernorm. allegeldy hurts interpretability. here we give ourselves the option to use it
        #insert is where we can insert our own attention or mlp values. this is for interpretability
        # assumes insert is a dictionary of {'inlayerpos': {pCL: value}}
        def replace(x, key):
            if key in insert:
                for CL in insert[key].keys():
                    print(key)
                    x[:, CL] = torch.tensor(insert[key][CL])
            return x
        hs = {}
        if layernorm:
            x = self.ln1(x)
        attn = self.attn(x)
        hs['attn'] = attn.clone().detach()
        replace(attn, 'attn')
        x = x + attn
        replace(x, 'attn-res')
        hs['attn-res'] = x.clone().detach()
        if layernorm:
            x = x + self.mlpf(self.ln2(x))

        mlpx = self.mlpf(x)
        hs['mlp'] = mlpx.clone().detach()
        replace(mlpx, 'mlp')
        x = x + mlpx
        replace(x, 'mlp-res')
        hs['mlp-res'] = x.clone().detach()
        if return_hs:
            return x, hs
        return x
    
    def get_attn(self, x, layernorm = False):
        if layernorm:
            x = self.ln1(x)
        return self.attn.get_attn(x)

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_layer = config.n_layer
        self.block = Block(config)
        self.n_embed = config.n_embd
        self.blocks = nn.ModuleList([Block(config) for _ in range(self.n_layer)])
        self.l_in = nn.Linear(config.in_dim, config.n_embd)
        self.l_out = nn.Linear(config.n_embd, config.out_dim)
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Initialize positional embeddings
        self.max_seq_length = config.max_seq_length
        self.positional_embeddings = nn.Parameter(torch.zeros(self.max_seq_length, self.n_embed))

        # report number of parameters 
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Total number of parameters: {total_params}')

    def forward(self, x, layernorm=False, insertall = {}):
        # Add positional embeddings
        seq_length = x.size(1)
        positions = torch.arange(0, seq_length, dtype=torch.long, device=x.device)
        if seq_length > self.max_seq_length:
            # add extra zeroes to positional embeddings
            diff = seq_length - self.max_seq_length
            extra_pos = torch.zeros(diff, self.n_embed, device=x.device)
            pos_embeddings = torch.cat((self.positional_embeddings, extra_pos), dim=0)
        
        pos_embeddings = self.positional_embeddings[positions]

        x = self.l_in(x)
        x = x + pos_embeddings  # Add positional embeddings to input embeddings

        for i in range(self.n_layer):
            insert = {}
            layer = i+1
            if layer in insertall:
                insert = insertall[layer]
            x = self.blocks[i](x, layernorm = layernorm, insert = insert)
        if layernorm:
            x = self.ln_f(x)
        y = self.l_out(x)
        return y

    def forward_hs(self, x, layernorm=False):
        # Add positional embeddings
        seq_length = x.size(1)
        positions = torch.arange(0, seq_length, dtype=torch.long, device=x.device)
        if seq_length > self.max_seq_length:
            # add extra zeroes to positional embeddings
            diff = seq_length - self.max_seq_length
            extra_pos = torch.zeros(diff, self.n_embed, device=x.device)
            pos_embeddings = torch.cat((self.positional_embeddings, extra_pos), dim=0)
            
        pos_embeddings = self.positional_embeddings[positions]

        x = self.l_in(x)
        x = x + pos_embeddings  # Add positional embeddings to input embeddings

        hidden_states = {0: {'inp': x}}

        for i in range(self.n_layer):
            x, hs= self.blocks[i](x, layernorm, return_hs = True)
            hidden_states[i+1] = hs
        if layernorm:
            x = self.ln_f(x)
        y = self.l_out(x)
        return y, hidden_states
    
    def return_attns(self, x, layernorm = False):
        # Add positional embeddings
        seq_length = x.size(1)
        positions = torch.arange(0, seq_length, dtype=torch.long, device=x.device)
        if seq_length > self.max_seq_length:
            # add extra zeroes to positional embeddings
            diff = seq_length - self.max_seq_length
            extra_pos = torch.zeros(diff, self.n_embed, device=x.device)
            pos_embeddings = torch.cat((self.positional_embeddings, extra_pos), dim=0)
            
        pos_embeddings = self.positional_embeddings[positions]

        x = self.l_in(x)
        x = x + pos_embeddings  # Add positional embeddings to input embeddings

        attns = []

        for i in range(self.n_layer):
            x = self.blocks[i](x, layernorm)
            attns.append(self.blocks[i].get_attn(x, layernorm))
        attns = torch.stack(attns)
        return attns
    
        







if __name__ == "__main__":
    #linspace between -100 and 100
    torch.manual_seed(0)
    config = get_default_config()
    config.n_embd = 16
    attn = CausalSelfAttention(config)
    model = Transformer(config)
    x = torch.rand(15000,10,2)
    y = model.return_attns(x)
    print(y.shape)

    

    # x = 
    # gelu = NewGELU()
    # y = gelu(x)
    # print(y)