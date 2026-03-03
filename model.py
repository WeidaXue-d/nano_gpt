import torch                            # type: ignore
import torch.nn as nn                   # type: ignore
from torch.nn import functional as F    # type: ignore

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




            

