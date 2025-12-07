# This file collects all the relevant code that we covered thus far
# throughout Chapters 2-4.
# This file can be run as a standalone script.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from konlpy.tag import Okt
import re

#####################################
# Chapter 2 - 한글/SentencePiece 버전
#####################################

# utils.py
from konlpy.tag import Okt
import re
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader

SPECIAL_TOKENS = ['<pad>', '<unk>', '<EOS>', '<PARA_END>', '<user>', '<meta>']

class OktTokenizer:
    def __init__(self, vocab_size=20000):
        self.okt = Okt()
        self.special_tokens = SPECIAL_TOKENS
        self.vocab_size = vocab_size
        self.vocab = None
        self.inv_vocab = None


    def preprocess(self, text):
        # 특수문자 분리
        text = re.sub(r'([.,!?~()\'\"/:;<>])', r' \1 ', text)
        # <EOS> 등은 단일 토큰으로 보존
        text = re.sub(r'\s*<EOS>\s*', r' <EOS> ', text)
        text = re.sub(r'\s*<PARA_END>\s*', r' <PARA_END> ', text)
        # 필요시 <PARA_END> 등도 동일 패턴
        text = re.sub(r'\s+', ' ', text)
        return text

    def tokenize(self, text, special_tokens=['<EOS>', '<PARA_END>', ...]):
        result = []
        for segment in text.split():
            if segment in special_tokens:
                result.append(segment)
            else:
                result.extend(self.okt.morphs(segment))
        return result


    def build_vocab(self, tokens):
        counter = Counter(tokens)
        vocab = {}
        idx = 0
        for tok in self.special_tokens:
            vocab[tok] = idx
            idx += 1
        for tok, _ in counter.most_common(self.vocab_size - len(self.special_tokens)):
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
        self.vocab = vocab
        self.inv_vocab = {idx:tok for tok, idx in vocab.items()}

    def encode(self, tokens, subword_tokenizer=None):
        """토큰 리스트 또는 단일 토큰을 ID 리스트로 변환"""
        if isinstance(tokens, str):
            tokens = [tokens]
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            elif subword_tokenizer is not None:
                sub_tokens = subword_tokenizer.tokenize(token)
                ids.extend([self.vocab.get(sub, self.vocab['<unk>']) for sub in sub_tokens])
            else:
                ids.append(self.vocab['<unk>'])
        return ids



    def decode(self, indices):
        if self.inv_vocab is None:
            raise RuntimeError('Vocab not built. Run build_vocab(tokens) first.')
        return [self.inv_vocab.get(idx, '<unk>') for idx in indices]

class GPTDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def create_okt_dataloader(ids, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True):
    inputs, targets = [], []
    for i in range(0, len(ids) - max_length, stride):
        input_chunk = ids[i:i + max_length]
        target_chunk = ids[i + 1:i + max_length + 1]
        inputs.append(torch.tensor(input_chunk))
        targets.append(torch.tensor(target_chunk))
    dataset = GPTDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader

def create_okt_dataloader_v2(ids, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, para_token_id=None):
    inputs, targets = [], []
    paragraphs = []
    if para_token_id is not None:
        start = 0
        for i, tok in enumerate(ids):
            if tok == para_token_id:
                paragraphs.append(ids[start:i+1])
                start = i + 1
        if start < len(ids):
            paragraphs.append(ids[start:])
    else:
        paragraphs = [ids]
    for para in paragraphs:
        if len(para) <= 1:
            continue
        for i in range(0, len(para) - max_length, stride):
            input_chunk = para[i:i + max_length]
            target_chunk = para[i + 1:i + max_length + 1]
            inputs.append(torch.tensor(input_chunk))
            targets.append(torch.tensor(target_chunk))
    dataset = GPTDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader

#####################################
# Chapter 3 - Multi-head Attention
#####################################

def multi_head_attention(Q, K, V, num_heads, mask_future=True, dropout_p=0.1):
    # Q, K, V shape: (batch, seq_len, emb_dim)
    batch_size, seq_len, emb_dim = Q.size()
    head_dim = emb_dim // num_heads
    
    # 1. 분할: head별로 나누기
    def split_heads(x):
        # (batch, seq_len, emb_dim) -> (batch, num_heads, seq_len, head_dim)
        return x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    Qh = split_heads(Q)
    Kh = split_heads(K)
    Vh = split_heads(V)
    
    # 2. 스케일 점수 계산
    scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32, device=Q.device))
    
    # 3. causal mask 적용 (상삼각)
    if mask_future:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask = mask.expand(batch_size, num_heads, seq_len, seq_len)
        scores = scores.masked_fill(mask, float('-inf'))
    
    # 4. softmax, dropout
    attn_weights = F.softmax(scores, dim=-1)
    attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)
    
    # 5. 어텐션 값 연산
    attn_output = torch.matmul(attn_weights, Vh)  # (batch, num_heads, seq_len, head_dim)
    
    # 6. head 합치기
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, emb_dim)
    
    return attn_output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim
        
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
        
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        
        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        
        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        
        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        
        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection
        
        return context_vec


#####################################
# Chapter 4 - GPT Model Components
#####################################

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
        
    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
        
    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        
        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        
        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :] 

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)
        
        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


if __name__ == "__main__":
    
    GPT_CONFIG_META = {
        "vocab_size": 6000,         # SentencePiece vocab size
        "context_length": 256,      # Context length
        "emb_dim": 256,            # Embedding dimension  
        "n_heads": 4,              # Number of attention heads
        "n_layers": 4,             # Number of layers
        "drop_rate": 0.1,          # Dropout rate
        "qkv_bias": False          # Query-Key-Value bias
    }
    
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_META)
    model.eval()  # disable dropout
    
    # 토크나이저와 텍스트 생성 예시
    tokenizer = SimpleKoreanTokenizer()
    start_context = "나는 열정 가득한 사람을"
    
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    
    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)
    
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor, 
        max_new_tokens=10,
        context_size=GPT_CONFIG_META["context_length"]
    )
    
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)