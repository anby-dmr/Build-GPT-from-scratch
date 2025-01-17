import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 16
block_size = 32
max_iters = 5000
eval_iters = 200
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embed_size = 64
num_heads = 4
num_layers = 4
dropout = 0.0

torch.manual_seed(1337)

# Load the data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Create vocab and encoder/decoder
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {character:index for index, character in enumerate(chars)}
itos = {index:character for index, character in enumerate(chars)}
encode = lambda chars: [stoi[x] for x in chars]
decode = lambda ints: [itos[x] for x in ints]

# Split the data
data = torch.tensor(encode(text), dtype=torch.long)
split_point = int(0.9 * len(data))
train_data = data[:split_point]
test_data = data[split_point:]

# Get a batch of data
def get_batch(split:str):
    if split == 'train':
        data = train_data
    else:
        data = test_data
    
    indices = torch.randint(len(data)-block_size, (batch_size, ))
    x = [data[start:start+block_size] for start in indices]
    y = [data[start+1:start+block_size+1] for start in indices]
    x, y = torch.stack(x), torch.stack(y)
    x, y = x.to(device), y.to(device)
    return x, y

# Attention! Head.
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.head_size = head_size

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        wei = q @ k.transpose(-1, -2) * self.head_size**(-0.5) # Guess this should be head_size instead of C according to the paper.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, value=-float('inf')) # Careful! Slice the tril.
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        
        v = self.value(x)
        b = wei @ v # (B, T, head_size)
        return b

# Attention! MultiHead.
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        b = torch.cat([h(x) for h in self.heads], dim=-1)
        b = self.proj(b)
        b = self.dropout(b)
        return b

# Attention! FeedForward.
class FeedForward(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.fc(x)

# Attention! Block.
class Block(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        head_size = embed_size // num_heads       
        self.att = MultiHeadAttention(num_heads, head_size)
        self.fc = FeedForward(embed_size)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
    
    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.fc(self.ln2(x))
        return x

# Attention! BigramModel.
class BigramModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(block_size, embed_size)
        self.blocks = nn.Sequential(*[Block(embed_size, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(self, x, target=None):
        B, T = x.shape

        tok_emb = self.token_embed(x) # (B, T, C)
        pos_emb = self.pos_embed(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_temp = logits.reshape(B*T, C)
            target_temp = target.reshape(B*T)
            loss = F.cross_entropy(logits_temp, target_temp)

        return logits, loss


    def generate(self, x, num_predict):
        for t in range(num_predict):
            x_cond = x[:,-block_size:] # (B, T)
            logits, _ = self(x_cond) # (B, T, vocab_size)
            logits = logits[:,-1,:] # (B, vocab_size)
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            x_next = torch.multinomial(probs, 1) # (B, 1)
            x = torch.cat([x, x_next], dim=1) # (B, T+1)
        return x

# Evaluate the loss
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval() # Careful!
    for split in ['train', 'eval']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item() # item(). No need for further tensor operation.
        out[split] = torch.mean(losses)
    model.train() # Careful!
    return out


model = BigramModel()
model = model.to(device)
# Print the num of parameters
print(sum(p.numel() for p in model.parameters())/1e6, 'M params')

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for it in range(max_iters):
    if it % eval_iters == 0 or it == max_iters - 1:
        losses = estimate_loss(model)
        print(f"Iter {it}, train loss: {losses['train']}, test loss: {losses['eval']}")
    
    X, Y = get_batch('train')
    logits, loss = model(X, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if it % 1000 == 0 or it == max_iters - 1:
        torch.save(model.state_dict(), 'GPT_dk_headSize' + str(it) + '.pth')