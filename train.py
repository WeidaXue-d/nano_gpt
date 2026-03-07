import torch # type: ignore
import os
from model import BigramLanguageModel

batch_size = 64  
block_size = 256


device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")


with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

model = BigramLanguageModel(vocab_size)
model.to(device)


checkpoint_path = 'shakespeare_gpt_v1.pth'
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Successfully loaded checkpoint! ")


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
 
    x, y = x.to(device), y.to(device) 
    return x, y

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

batch_size = 64 
for steps in range(30000):
    xb, yb = get_batch('train')

   
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    
    if steps % 500 == 0:
        print(f"step: {steps} loss: {loss.item():.4f}")
        torch.save(model.state_dict(), checkpoint_path)


context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

'''
import torch # type: ignore
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

model = BigramLanguageModel(vocab_size)
model.to(device)

checkpoint_path = 'shakespeare_gpt_v1.pth'
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"loaded checkpoint ")

from model import BigramLanguageModel

with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()
#data size
print(f'total words:{len(text)}')

#vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f'size of vocab: {vocab_size}')

#mapping
# stoi: string to integer
stoi = { ch:i for i,ch in enumerate(chars)}
# itos: integer to string
itos = { i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
    
#tensor
data = torch.tensor(encode(text), dtype = torch.long)
print(f"shape of data: {data.shape}")

#training & verification set
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

batch_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split == 'train' else val_data

    #random start
    ix = torch.randint(len(data)-block_size,(batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x,y

xb , yb = get_batch('train')

print(f'(Batch,time): {xb.shape}')
print('input x')
print(xb)
print('expect y')
print(yb)

for t in range(block_size):
    context = xb[0, :t+1]
    target = yb[0,t]
    print(f'input: {context}, target: {target}')

m = BigramLanguageModel(vocab_size)
model = BigramLanguageModel(vocab_size)
model.to(device)

import os
checkpoint_path = 'shakespeare_gpt_v1.pth'
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Successfully loaded checkpoint from {checkpoint_path}! ")


logits, loss = m(xb, yb)

print(f"(B, T, C): {logits.shape}")
print(f"loss: {loss}")

context = torch.zeros((1,1), dtype = torch.long)
print(decode(m.generate(context, max_new_tokens = 100)[0].tolist()))

optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-4) #learning rate 0.001

batch_size = 64

for steps in range(30000):
    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    
    #Backward
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()
    if steps % 100 == 0:
        print(f"step: {steps} loss: {loss.item():.4f}")

context = torch.zeros((1,1), dtype = torch.long)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
'''