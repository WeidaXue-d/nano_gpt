import torch # type: ignore

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

