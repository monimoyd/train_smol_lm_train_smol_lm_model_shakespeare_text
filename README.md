# train_smol_lm_train_smol_lm_model_shakespeare_text

## SmolLM2-135M Model

SmolLM2-135M is a lightweight Transformer model designed for sequence prediction tasks. 
    Key components:
    - Embedding Layer: Encodes token indices into dense vectors of size `embed_dim`.
    - Positional Embeddings: Adds position information to token embeddings.
    - Transformer Encoder Layers: Stacks multiple self-attention layers with residual connections.
    - Output Layer: Maps the final hidden states to vocabulary logits.

    Parameters:
    - vocab_size: Size of the vocabulary.
    - embed_dim: Dimension of token embeddings.
    - num_heads: Number of attention heads.
    - num_layers: Number of Transformer encoder layers.
    - max_seq_len: Maximum sequence length.


    Model architecture is as below:

    OptimizedModule(
  (_orig_mod): SmolLM(
    (embedding): Embedding(50257, 512)
    (layers): ModuleList(
      (0-3): 4 x TransformerEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
        )
        (linear1): Linear(in_features=512, out_features=2048, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=2048, out_features=512, bias=True)
        (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
      )
    )
    (fc_out): Linear(in_features=512, out_features=50257, bias=True)
  )
)

Model has 64,188,497 trainable parameters.

## Techniques used during training
In  this repository I have trained SmolLM2-135 model. The following techniques are used:

* set_float32_matmul_precision
*  Torch.compile for optimization
* Autocast
* Training it for 5000 steps while predicting every 500 steps on what it utters. Now fully stop the model and save a checkpoint. Now load this checkpoint and train for 50 more steps

  ## Training Logs

  

from google.colab import drive
drive.mount('/content/drive')

Mounted at /content/drive

! cp /content/drive/MyDrive/session_12/shakespeare_drama.txt .

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os

# Define the SmolLM2-135M model (a simplified version of a Transformer)
class SmolLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len):
        super(SmolLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_embedding[:, :seq_len, :]
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)
        
    def parameter_count(self):
        """Calculates the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Shakespeare dataset
class ShakespeareDataset(Dataset):
    def __init__(self, tokenizer, text, seq_len):
        self.tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True).input_ids[0]
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        return (
            self.tokens[idx:idx + self.seq_len],
            self.tokens[idx + 1:idx + self.seq_len + 1]
        )

# Training parameters
embed_dim = 512
num_heads = 8
num_layers = 4
max_seq_len = 128
vocab_size = 50257
batch_size = 16
initial_steps = 5000
resume_steps = 50
eval_interval = 500

# Optimizations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")

# Load tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

text = open("shakespeare_drama.txt").read()  # Load Shakespeare text
train_dataset = ShakespeareDataset(tokenizer, text, max_seq_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = SmolLM(vocab_size, embed_dim, num_heads, num_layers, max_seq_len).to(device)
optimizer = AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
model = torch.compile(model)  # Torch.compile for optimization

print(f"Model has {model.parameter_count():,} trainable parameters.")

Model has 64,188,497 trainable parameters.

print(model)

OptimizedModule(
  (_orig_mod): SmolLM(
    (embedding): Embedding(50257, 512)
    (layers): ModuleList(
      (0-3): 4 x TransformerEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
        )
        (linear1): Linear(in_features=512, out_features=2048, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=2048, out_features=512, bias=True)
        (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
      )
    )
    (fc_out): Linear(in_features=512, out_features=50257, bias=True)
  )
)

# Training loop
def train_model(model, optimizer, criterion, dataloader, steps, eval_interval, checkpoint_path, start_step=1):
    model.train()
    progress = tqdm(total=steps, desc="Training", initial=start_step - 1)
    step = start_step - 1
    for epoch in range(steps // len(dataloader) + 1):
        for batch_idx, (x, y) in enumerate(dataloader):
            if step >= steps:
                break

            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):  # Autocast
                outputs = model(x)
                loss = criterion(outputs.view(-1, vocab_size), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            progress.update(1)

            if step % eval_interval == 0:
                print(f"Step {step}: Loss = {loss.item():.4f}")
                eval_model(model, tokenizer, device)

            if step >= steps:
                break

    # Save checkpoint
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

# Evaluation
@torch.no_grad()
def eval_model(model, tokenizer, device, prompt="To be, or not to be"):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = model(input_ids)
    predictions = torch.argmax(outputs, dim=-1)
    decoded = tokenizer.decode(predictions[0], skip_special_tokens=True)
    print(f"Model utterance: {decoded}")

Training:  10%|█         | 500/5000 [01:08<05:14, 14.30it/s]

Step 500: Loss = 0.7398
Model utterance: , partly
, arms them partly

Training:  20%|██        | 1002/5000 [01:47<05:06, 13.02it/s]

Step 1000: Loss = 0.4175
Model utterance: 
 partly
, arms fam partly

Training:  30%|███       | 1502/5000 [02:25<04:25, 13.18it/s]

Step 1500: Loss = 0.2951
Model utterance: 
 barren
, arms them barren

Training:  40%|████      | 2002/5000 [03:02<03:45, 13.27it/s]

Step 2000: Loss = 0.1800
Model utterance: 
 barren
, arms them barren

Training:  50%|█████     | 2502/5000 [03:39<03:11, 13.04it/s]

Step 2500: Loss = 0.1989
Model utterance: 
 barren
, arms them barren

Training:  60%|██████    | 3002/5000 [04:17<02:30, 13.30it/s]

Step 3000: Loss = 0.1243
Model utterance: 
 barren
, arms them barren

Training:  70%|███████   | 3502/5000 [04:54<01:52, 13.35it/s]

Step 3500: Loss = 0.0814
Model utterance: 
 barren
, arms them barren

Training:  80%|████████  | 4002/5000 [05:32<01:15, 13.17it/s]

Step 4000: Loss = 0.0918
Model utterance: 
 barren

 arms them barren

Training:  90%|█████████ | 4502/5000 [06:09<00:37, 13.32it/s]

Step 4500: Loss = 0.0767
Model utterance: en barren
, arms them barren

Training: 100%|██████████| 5000/5000 [06:46<00:00, 13.40it/s]

Step 5000: Loss = 0.0626
Model utterance: 
 barren
, arms them barren
Checkpoint saved to smollm_checkpoint.pth

<ipython-input-10-cadf5e4e1438>:6: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(checkpoint_path))

Checkpoint loaded. Resuming training...


Training:  99%|█████████▉| 5000/5050 [00:00<?, ?it/s]

Training:  99%|█████████▉| 5001/5050 [00:00<00:05,  8.51it/s]

Training:  99%|█████████▉| 5003/5050 [00:00<00:03, 11.80it/s]

Training:  99%|█████████▉| 5005/5050 [00:00<00:03, 12.37it/s]

Training:  99%|█████████▉| 5007/5050 [00:00<00:03, 12.85it/s]

Training:  99%|█████████▉| 5009/5050 [00:00<00:03, 12.98it/s]

Training:  99%|█████████▉| 5011/5050 [00:00<00:02, 13.05it/s]

Training:  99%|█████████▉| 5013/5050 [00:01<00:02, 13.00it/s]

Training:  99%|█████████▉| 5015/5050 [00:01<00:02, 12.99it/s]

Training:  99%|█████████▉| 5017/5050 [00:01<00:02, 13.05it/s]

Training:  99%|█████████▉| 5019/5050 [00:01<00:02, 13.08it/s]

Training:  99%|█████████▉| 5021/5050 [00:01<00:02, 13.13it/s]

Training:  99%|█████████▉| 5023/5050 [00:01<00:02, 13.12it/s]

Training: 100%|█████████▉| 5025/5050 [00:01<00:01, 13.00it/s]

Training: 100%|█████████▉| 5027/5050 [00:02<00:01, 13.05it/s]

Training: 100%|█████████▉| 5029/5050 [00:02<00:01, 13.09it/s]

Training: 100%|█████████▉| 5031/5050 [00:02<00:01, 12.95it/s]

Training: 100%|█████████▉| 5033/5050 [00:02<00:01, 13.15it/s]

Training: 100%|█████████▉| 5035/5050 [00:02<00:01, 13.16it/s]

Training: 100%|█████████▉| 5037/5050 [00:02<00:00, 13.13it/s]

Training: 100%|█████████▉| 5039/5050 [00:03<00:00, 13.07it/s]

Training: 100%|█████████▉| 5041/5050 [00:03<00:00, 13.08it/s]

Training: 100%|█████████▉| 5043/5050 [00:03<00:00, 13.11it/s]

Training: 100%|█████████▉| 5045/5050 [00:03<00:00, 13.11it/s]

Training: 100%|█████████▉| 5047/5050 [00:03<00:00, 13.03it/s]

Training: 100%|██████████| 5050/5050 [00:04<00:00, 10.92it/s]

Checkpoint saved to final_checkpoint.pth
Training complete.

  

  


