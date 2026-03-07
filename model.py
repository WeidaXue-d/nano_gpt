import torch                            # type: ignore
import torch.nn as nn                   # type: ignore
from torch.nn import functional as F    # type: ignore

batch_size = 64
block_size = 256  
n_embd = 384      
n_head = 6       
n_layer = 6       
dropout = 0.2     

#B, T, C = 4, 8, 2
#tril = torch.tril(torch.ones(T, T))
#wei = torch.zeros((T, T))
#wei = wei.masked_fill(tril == 0, float('-inf'))
#wei = F.softmax(wei, dim=-1)

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        #Q K V
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        #Q K 
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1]**-0.5)

        #Masking
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf') )
        wei = F.softmax(wei, dim=-1) #(B, T, T)
        wei = self.dropout(wei)

        v = self.value(x) #(B, T, head_size)
        out = wei @ v
        return out


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        #embedding
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd )

        #position embedding
        #location to Transformer
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        #self.sa_heads = MultiHeadAttention(4, n_embd //4) 
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)


    def forward(self, idx, targets = None):
        B, T = idx.shape

        #(B, T, n_embd)
        tok_emb = self.token_embedding_table(idx)
        #(T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))

        x = tok_emb + pos_emb   #(B, T, n_embd)

        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x) #(B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)


            logits = logits[:, -1, :]
            #(B, C)
            probs = F.softmax(logits, dim = -1)
            #(B, 1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            #(B, T+1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx 
    

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super(). __init__()

        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x):
        #（B, T, head_size) * num_heads -> (B, T, n_embd)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        #Layer Normalization
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
            




'''
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):
        #(Batch, Time) -> (Batch=4, Time=8, Channel=65)
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            #Reshaping
            B, T, C = logits.shape

            logits = logits.view(B*T, C)    #(32,65)
            targets = targets.view(B*T)     #(32)

            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            #(B, C)
            probs = F.softmax(logits, dim = -1)
            #(B, 1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            #(B, T+1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx 
        '''