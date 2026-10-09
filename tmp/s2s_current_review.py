# CELL 2
import importlib.util

required_packages = {
    "torch": "torch>=2.2",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm",
    "sacrebleu": "sacrebleu",
}
missing_packages = [
    package for module, package in required_packages.items()
    if importlib.util.find_spec(module) is None
]
if missing_packages:
    get_ipython().run_line_magic("pip", "install " + " ".join(missing_packages))
else:
    print("Đã có đủ thư viện.")


# CELL 3
import csv
import hashlib
import html
import json
import math
import random
import re
import time
import unicodedata
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn.utils.rnn import (
    pack_padded_sequence, pad_packed_sequence, pad_sequence,
)
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sacrebleu.metrics import BLEU, CHRF


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# CELL 4
# Có thể sửa các giá trị này rồi Restart Kernel -> Run All.
# Kaggle: bật GPU trong Settings nếu muốn train bằng CUDA.
CFG = {
    "seed": 42,
    "max_train_samples": 5_000,
    "max_validation_samples": 500,
    "max_test_samples": 500,
    "validation_fraction": 0.05,
    "max_src_tokens": 50,
    "max_tgt_tokens": 50,
    "src_vocab_size": 12_000,
    "tgt_vocab_size": 12_000,
    "min_freq": 2,
    "embedding_dim": 128,
    "hidden_dim": 128,
    "dropout": 0.2,
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 1e-3,
    "gradient_clip": 1.0,
    "patience": 3,
    "cpu_threads": 4,
}

assert CFG["epochs"] >= 1 and CFG["batch_size"] >= 1
assert 0 < CFG["validation_fraction"] < 1
assert CFG["min_freq"] >= 1 and CFG["patience"] >= 1
assert min(CFG["max_src_tokens"], CFG["max_tgt_tokens"]) >= 1
assert min(CFG["src_vocab_size"], CFG["tgt_vocab_size"]) >= 5
for name in ("max_train_samples", "max_validation_samples", "max_test_samples"):
    assert CFG[name] is None or CFG[name] > 0, name


# Để None: ưu tiên đường dẫn dự án Kaggle, rồi tìm dữ liệu local/Kaggle Input.
# Muốn chỉ định thư mục khác: thay None bằng Path("/duong/dan/data").
DATA_DIR_OVERRIDE = None
KAGGLE_DATA_DIR = Path("/kaggle/working/Encode_Decode_NLP/data")


def find_data_dir(data_dir_override=None):
    """Tìm đủ 3 TSV, ưu tiên thư mục dự án đã đặt trong Kaggle Working."""
    required = ("train.tsv", "validation.tsv", "test.tsv")

    def missing_files(directory):
        return [name for name in required if not (directory / name).is_file()]

    if data_dir_override is not None:
        directory = Path(data_dir_override).expanduser()
        missing = missing_files(directory)
        if missing:
            raise FileNotFoundError(
                f"Thư mục dữ liệu {directory} thiếu: {', '.join(missing)}. "
                "Kiểm tra tên file hoặc sửa DATA_DIR_OVERRIDE."
            )
        return directory.resolve()

    candidates = [KAGGLE_DATA_DIR]
    for root in (Path.cwd(), *Path.cwd().parents):
        candidates.extend((root / "data", root))
    for directory in candidates:
        if not missing_files(directory):
            return directory.resolve()

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for train_file in sorted(kaggle_input.rglob("train.tsv")):
            if not missing_files(train_file.parent):
                return train_file.parent.resolve()
    return None


DATA_DIR = find_data_dir(DATA_DIR_OVERRIDE)
if DATA_DIR is None:
    missing = [name for name in ("train.tsv", "validation.tsv", "test.tsv")
               if not (KAGGLE_DATA_DIR / name).is_file()]
    raise FileNotFoundError(
        f"Không tìm thấy đủ 3 file TSV. Đường dẫn ưu tiên: {KAGGLE_DATA_DIR}; "
        f"thiếu: {', '.join(missing)}. "
        "Đặt dữ liệu tại thư mục này hoặc sửa DATA_DIR_OVERRIDE. "
        "Notebook cũng tìm trong data/ local và /kaggle/input."
    )


PROJECT_DIR = DATA_DIR.parent
WORK_DIR = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else PROJECT_DIR
RUN_DIR = (
    WORK_DIR / "outputs" / "s2s"
    / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
)
RUN_DIR.mkdir(parents=True, exist_ok=True)

set_seed(CFG["seed"])
torch.set_num_threads(CFG["cpu_threads"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch:", torch.__version__, "| Thiết bị:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
print("Data:", DATA_DIR)
print("Kết quả:", RUN_DIR)
print(json.dumps(CFG, ensure_ascii=False, indent=2))


# CELL 6
TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)

def normalize_text(text):
    text = unicodedata.normalize("NFKC", html.unescape(text)).lower()
    return " ".join(text.split())

def tokenize(text):
    return TOKEN_PATTERN.findall(normalize_text(text))

def tokens_to_text(tokens):
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    return re.sub(r"([(\[])\s+", r"\1", text)

def read_tsv(path):
    pairs, seen = [], set()
    stats = {"rows": 0, "duplicates": 0, "empty": 0, "too_long": 0}
    # QUOTE_NONE: dấu " trong câu là nội dung, không phải quote của TSV.
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        for line_number, row in enumerate(reader, start=1):
            stats["rows"] += 1
            if len(row) != 2:
                raise ValueError(f"{path.name}:{line_number}: cần đúng 2 cột.")
            src_tokens, tgt_tokens = tokenize(row[0]), tokenize(row[1])
            if not src_tokens or not tgt_tokens:
                stats["empty"] += 1
                continue
            if (len(src_tokens) > CFG["max_src_tokens"]
                    or len(tgt_tokens) > CFG["max_tgt_tokens"]):
                stats["too_long"] += 1
                continue
            pair = (" ".join(src_tokens), " ".join(tgt_tokens))
            if pair in seen:
                stats["duplicates"] += 1
                continue
            seen.add(pair)
            pairs.append(pair)
    if not pairs:
        raise ValueError(f"{path.name} không có cặp hợp lệ sau lọc.")
    return pairs, stats


# CELL 7
raw_pairs, data_stats = {}, {}
for split in ("train", "validation", "test"):
    path = DATA_DIR / f"{split}.tsv"
    if not path.is_file():
        raise FileNotFoundError(path)
    raw_pairs[split], data_stats[split] = read_tsv(path)
    print(f"{split:10s} | {data_stats[split]} | giữ {len(raw_pairs[split]):,} cặp")

def source_set(pairs):
    return {src for src, _ in pairs}

def sample_pairs(pairs, limit, seed):
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    return pairs if limit is None else pairs[:limit]

def split_by_source(pairs, fraction, seed):
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair[0]].append(pair)
    sources = list(groups)
    random.Random(seed).shuffle(sources)
    if len(sources) < 2:
        raise ValueError("Cần ít nhất 2 câu nguồn khác nhau để chia train/validation.")
    desired_count = max(1, round(len(pairs) * fraction))
    val_sources, count = set(), 0
    # Luôn giữ ít nhất một nhóm cho train.
    for src in sources[:-1]:
        val_sources.add(src)
        count += len(groups[src])
        if count >= desired_count:
            break
    train = [pair for pair in pairs if pair[0] not in val_sources]
    validation = [pair for pair in pairs if pair[0] in val_sources]
    return train, validation

test_sources = source_set(raw_pairs["test"])
overlap_val_test = source_set(raw_pairs["validation"]) & test_sources
removed_test_overlap = sum(
    src in test_sources for src, _ in raw_pairs["train"]
)
# Dùng toàn bộ test sau lọc để loại overlap, trước khi giới hạn số mẫu test.
train_pool = [
    pair for pair in raw_pairs["train"] if pair[0] not in test_sources
]

validation_rebuilt = bool(overlap_val_test)
if validation_rebuilt:
    print(
        f"\nValidation/test trùng {len(overlap_val_test):,} câu nguồn. "
        "Tạo validation mới từ train."
    )
    train_pool, validation_pool = split_by_source(
        train_pool, CFG["validation_fraction"], CFG["seed"]
    )
    removed_validation_overlap = 0
else:
    validation_pool = raw_pairs["validation"]
    validation_sources = source_set(validation_pool)
    removed_validation_overlap = sum(
        src in validation_sources for src, _ in train_pool
    )
    train_pool = [
        pair for pair in train_pool if pair[0] not in validation_sources
    ]

train_pairs = sample_pairs(train_pool, CFG["max_train_samples"], CFG["seed"])
validation_pairs = sample_pairs(
    validation_pool, CFG["max_validation_samples"], CFG["seed"] + 1
)
test_pairs = sample_pairs(
    raw_pairs["test"], CFG["max_test_samples"], CFG["seed"] + 2
)

for split, pairs in (
    ("train", train_pairs), ("validation", validation_pairs), ("test", test_pairs)
):
    if not pairs:
        raise ValueError(f"Tập {split} rỗng, hãy tăng giới hạn độ dài/số mẫu.")

assert source_set(train_pairs).isdisjoint(source_set(validation_pairs))
assert source_set(train_pairs).isdisjoint(source_set(test_pairs))
assert source_set(validation_pairs).isdisjoint(source_set(test_pairs))

split_summary = {
    "input_stats": data_stats,
    "validation_rebuilt_from_train": validation_rebuilt,
    "removed_train_pairs_with_test_source": removed_test_overlap,
    "removed_train_pairs_with_validation_source": removed_validation_overlap,
    "selected_pairs": {
        "train": len(train_pairs),
        "validation": len(validation_pairs),
        "test": len(test_pairs),
    },
    "file_sha256": {
        split: hashlib.sha256((DATA_DIR / f"{split}.tsv").read_bytes()).hexdigest()
        for split in ("train", "validation", "test")
    },
}
print("Loại cặp train trùng câu nguồn test:", removed_test_overlap)
print("Số cặp sử dụng:", split_summary["selected_pairs"])
print("\nMột ví dụ train:")
print("EN:", train_pairs[0][0])
print("VI:", train_pairs[0][1])


# CELL 9
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
PAD, UNK, BOS, EOS = range(4)

class Vocabulary:
    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        assert self.itos[:4] == SPECIAL_TOKENS

    @classmethod
    def build(cls, token_sequences, max_size, min_freq):
        counts = Counter(token for sequence in token_sequences for token in sequence)
        # Tie-break theo chữ để mapping ổn định.
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        tokens = [
            token for token, count in ranked
            if count >= min_freq and token not in SPECIAL_TOKENS
        ][:max_size - len(SPECIAL_TOKENS)]
        return cls(SPECIAL_TOKENS + tokens)

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(token, UNK) for token in tokens]

    def decode(self, ids):
        tokens = []
        for idx in ids:
            if idx == EOS:
                break
            if idx not in (PAD, BOS):
                tokens.append(self.itos[idx])
        return tokens


class TranslationDataset(Dataset):
    def __init__(self, pairs, source_vocab, target_vocab):
        self.pairs = list(pairs)
        self.encoded = [
            (
                torch.tensor(source_vocab.encode(src.split()) + [EOS]),
                torch.tensor([BOS] + target_vocab.encode(tgt.split()) + [EOS]),
            )
            for src, tgt in self.pairs
        ]

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, index):
        return self.encoded[index]

def collate_batch(batch):
    sources, targets = zip(*batch)
    lengths = torch.tensor([len(src) for src in sources], dtype=torch.long)
    return (
        pad_sequence(sources, batch_first=True, padding_value=PAD),
        lengths,  # pack_padded_sequence cần lengths trên CPU.
        pad_sequence(targets, batch_first=True, padding_value=PAD),
    )

def unknown_rate(pairs, column, vocab):
    tokens = [token for pair in pairs for token in pair[column].split()]
    return sum(token not in vocab.stoi for token in tokens) / max(len(tokens), 1)


# CELL 10
src_vocab = Vocabulary.build(
    (src.split() for src, _ in train_pairs),
    CFG["src_vocab_size"], CFG["min_freq"],
)
tgt_vocab = Vocabulary.build(
    (tgt.split() for _, tgt in train_pairs),
    CFG["tgt_vocab_size"], CFG["min_freq"],
)


train_dataset = TranslationDataset(train_pairs, src_vocab, tgt_vocab)
validation_dataset = TranslationDataset(validation_pairs, src_vocab, tgt_vocab)
test_dataset = TranslationDataset(test_pairs, src_vocab, tgt_vocab)

loader_generator = torch.Generator().manual_seed(CFG["seed"])
loader_options = {
    "batch_size": CFG["batch_size"],
    "collate_fn": collate_batch,
    "num_workers": 0,
    "pin_memory": DEVICE.type == "cuda",
}
train_loader = DataLoader(
    train_dataset, shuffle=True, generator=loader_generator, **loader_options
)
validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

print(f"Vocabulary: EN={len(src_vocab):,} | VI={len(tgt_vocab):,}")
for split, pairs in (("validation", validation_pairs), ("test", test_pairs)):
    print(
        f"{split}: EN <unk>={unknown_rate(pairs, 0, src_vocab):.2%} | "
        f"VI <unk>={unknown_rate(pairs, 1, tgt_vocab):.2%}"
    )
sample_src, sample_lengths, sample_tgt = collate_batch(
    [train_dataset[i] for i in range(min(3, len(train_dataset)))]
)
print("Một batch: src", tuple(sample_src.shape), "| target", tuple(sample_tgt.shape))
print("Decoder input:", sample_tgt[0, :-1].tolist())
print("Target labels:", sample_tgt[0, 1:].tolist())


# CELL 12
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.memory_projection = nn.Linear(2 * hidden_dim, hidden_dim)
        self.hidden_bridge = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, src, lengths):
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, hidden = self.gru(packed)
        output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=src.size(1)
        )
        memory = torch.tanh(self.memory_projection(output))
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        initial_decoder_hidden = torch.tanh(self.hidden_bridge(last_hidden)).unsqueeze(0)
        return memory, initial_decoder_hidden, src.ne(PAD)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.query_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fusion = nn.Linear(2 * hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, input_ids, hidden, memory, source_mask):
        embedded = self.dropout(self.embedding(input_ids))
        states, hidden = self.gru(embedded, hidden)
        scores = torch.bmm(
            self.query_projection(states), memory.transpose(1, 2)
        ) / self.scale
        scores = scores.masked_fill(~source_mask.unsqueeze(1), float("-inf"))
        attention = scores.softmax(dim=-1)
        context = torch.bmm(attention, memory)
        fused = torch.tanh(self.fusion(torch.cat((states, context), dim=-1)))
        logits = self.output_projection(self.dropout(fused))
        return logits, hidden, attention

class Seq2Seq(nn.Module):
    def __init__(self, src_size, tgt_size, embedding_dim, hidden_dim, dropout):
        super().__init__()
        self.encoder = Encoder(src_size, embedding_dim, hidden_dim, dropout)
        self.decoder = Decoder(tgt_size, embedding_dim, hidden_dim, dropout)

    def forward(self, src, lengths, decoder_inputs):
        memory, hidden, mask = self.encoder(src, lengths)
        logits, _, attention = self.decoder(decoder_inputs, hidden, memory, mask)
        return logits, attention

def make_model(config, source_vocab, target_vocab):
    return Seq2Seq(
        len(source_vocab), len(target_vocab),
        config["embedding_dim"], config["hidden_dim"], config["dropout"],
    )


# CELL 13
set_seed(CFG["seed"])
model = make_model(CFG, src_vocab, tgt_vocab).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=PAD, reduction="sum")

# Một forward để kiểm tra shape và mask trước khi train.
model.eval()
with torch.no_grad():
    sample_logits, sample_attention = model(
        sample_src.to(DEVICE), sample_lengths, sample_tgt[:, :-1].to(DEVICE)
    )
    assert sample_logits.shape == (
        sample_src.size(0), sample_tgt.size(1) - 1, len(tgt_vocab)
    )
    assert torch.isfinite(sample_logits).all()
    assert torch.allclose(
        sample_attention.sum(-1), torch.ones_like(sample_attention.sum(-1)), atol=1e-5
    )
    padding_mask = sample_src.eq(PAD).to(DEVICE).unsqueeze(1).expand_as(sample_attention)
    assert torch.all(sample_attention.masked_select(padding_mask) == 0)

print(f"Số tham số: {sum(p.numel() for p in model.parameters()):,}")
print("Logits:", tuple(sample_logits.shape), "| Attention:", tuple(sample_attention.shape))
print("Forward và mask hợp lệ.")


# CELL 15
def run_epoch(loader, optimizer=None, description=""):
    training = optimizer is not None
    model.train(training)
    total_loss, total_tokens, correct_tokens = 0.0, 0, 0
    progress = tqdm(loader, desc=description, leave=False)
    for src, lengths, tgt in progress:
        src = src.to(DEVICE, non_blocking=True)
        tgt = tgt.to(DEVICE, non_blocking=True)
        decoder_inputs, labels = tgt[:, :-1], tgt[:, 1:]
        valid_tokens = labels.ne(PAD)
        token_count = int(valid_tokens.sum().item())
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits, _ = model(src, lengths, decoder_inputs)
            loss_sum = criterion(logits.reshape(-1, len(tgt_vocab)), labels.reshape(-1))
            loss = loss_sum / token_count
            if not torch.isfinite(loss):
                raise RuntimeError("Loss không hữu hạn, thử giảm learning_rate.")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CFG["gradient_clip"])
                optimizer.step()
        total_loss += float(loss_sum.detach().item())
        total_tokens += token_count
        correct_tokens += int(
            ((logits.argmax(-1) == labels) & valid_tokens).sum().item()
        )
        progress.set_postfix(loss=f"{total_loss / total_tokens:.3f}")
    return {
        "loss": total_loss / total_tokens,
        "token_accuracy": correct_tokens / total_tokens,
        "tokens": total_tokens,
    }

def write_json(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

write_json(RUN_DIR / "config.json", CFG)
write_json(RUN_DIR / "data_summary.json", split_summary)
write_json(
    RUN_DIR / "vocab.json", {"src_itos": src_vocab.itos, "tgt_itos": tgt_vocab.itos}
)

optimizer = torch.optim.Adam(model.parameters(), lr=CFG["learning_rate"])
BEST_PATH = RUN_DIR / "best_model.pt"
history = []
best_val_loss, best_epoch, stale_epochs = float("inf"), 0, 0

for epoch in range(1, CFG["epochs"] + 1):
    start = time.perf_counter()
    train_metrics = run_epoch(train_loader, optimizer, f"Train {epoch}/{CFG['epochs']}")
    val_metrics = run_epoch(validation_loader, description="Validation")
    elapsed = time.perf_counter() - start
    improved = val_metrics["loss"] < best_val_loss
    if improved:
        best_val_loss, best_epoch, stale_epochs = val_metrics["loss"], epoch, 0
        checkpoint = {
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "config": dict(CFG),
            "src_itos": src_vocab.itos,
            "tgt_itos": tgt_vocab.itos,
            "epoch": epoch,
            "validation_loss": best_val_loss,
            "data_summary": split_summary,
            "torch_version": str(torch.__version__),
        }
        torch.save(checkpoint, BEST_PATH)
    else:
        stale_epochs += 1
    history.append({
        "epoch": epoch,
        "train_loss": train_metrics["loss"],
        "validation_loss": val_metrics["loss"],
        "train_token_accuracy": train_metrics["token_accuracy"],
        "validation_token_accuracy": val_metrics["token_accuracy"],
        "seconds": elapsed,
    })
    write_json(RUN_DIR / "history.json", history)
    print(
        f"Epoch {epoch:02d} | train loss={train_metrics['loss']:.4f} | "
        f"val loss={val_metrics['loss']:.4f} | "
        f"val token acc={val_metrics['token_accuracy']:.2%} | "
        f"{elapsed:.1f}s" + (" | lưu best" if improved else "")
    )
    if stale_epochs >= CFG["patience"]:
        print("Early stopping.")
        break

print(f"\nBest epoch: {best_epoch} | validation loss: {best_val_loss:.4f}")
print("Checkpoint:", BEST_PATH)


# CELL 16
if not BEST_PATH.is_file():
    raise FileNotFoundError("Chưa có checkpoint. Hãy train ít nhất 1 epoch hoàn chỉnh.")

best_checkpoint = torch.load(BEST_PATH, map_location="cpu", weights_only=True)
model.load_state_dict(best_checkpoint["model_state"])
model.to(DEVICE).eval()
print("Đã lấy model tại epoch:", best_checkpoint["epoch"])

epochs = [row["epoch"] for row in history]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(epochs, [r["train_loss"] for r in history], "o-", label="Train")
axes[0].plot(epochs, [r["validation_loss"] for r in history], "o-", label="Validation")
axes[0].set(xlabel="Epoch", ylabel="Cross-entropy / token", title="Loss (teacher forcing)")
axes[1].plot(
    epochs, [r["validation_token_accuracy"] for r in history], "o-", color="tab:green"
)
axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Validation token accuracy")
axes[0].legend()
plt.tight_layout()
plt.show()


# CELL 18
@torch.inference_mode()
def greedy_decode(
    texts, current_model, source_vocab, target_vocab,
    max_source_tokens, max_new_tokens, return_attention=False,
):
    if not texts:
        return ([], [], None) if return_attention else []
    current_model.eval()
    device = next(current_model.parameters()).device
    token_lists = [tokenize(text) for text in texts]
    if any(not tokens for tokens in token_lists):
        raise ValueError("Câu đầu vào cần có ít nhất một token.")
    if any(len(tokens) > max_source_tokens for tokens in token_lists):
        raise ValueError(
            f"Câu đầu vào vượt {max_source_tokens} token. "
            "Hãy chia câu hoặc tăng max_source_tokens."
        )
    sources = [
        torch.tensor(source_vocab.encode(tokens) + [EOS], dtype=torch.long)
        for tokens in token_lists
    ]
    lengths = torch.tensor([len(src) for src in sources], dtype=torch.long)
    src = pad_sequence(sources, batch_first=True, padding_value=PAD).to(device)
    memory, hidden, source_mask = current_model.encoder(src, lengths)
    next_input = torch.full((len(texts), 1), BOS, dtype=torch.long, device=device)
    finished = torch.zeros(len(texts), dtype=torch.bool, device=device)
    generated, attention_steps = [], []
    for _ in range(max_new_tokens):
        logits, hidden, attention = current_model.decoder(
            next_input, hidden, memory, source_mask
        )
        scores = logits[:, -1].clone()
        scores[:, [PAD, BOS]] = float("-inf")
        next_ids = scores.argmax(-1)
        next_ids = torch.where(finished, torch.full_like(next_ids, PAD), next_ids)
        generated.append(next_ids.cpu())
        if return_attention:
            attention_steps.append(attention[:, 0].cpu())
        finished |= next_ids.eq(EOS)
        next_input = next_ids.unsqueeze(1)
        if bool(finished.all()):
            break
    generated = torch.stack(generated, dim=1).tolist()
    translations = [
        tokens_to_text(target_vocab.decode(ids)) for ids in generated
    ]
    if return_attention:
        attention_matrix = torch.stack(attention_steps, dim=1).numpy()
        source_labels = [tokens + ["<eos>"] for tokens in token_lists]
        return translations, (generated, source_labels), attention_matrix
    return translations


# CELL 19
test_metrics = run_epoch(test_loader, description="Test loss")
print(
    f"Test loss (teacher forcing)={test_metrics['loss']:.4f} | "
    f"perplexity={math.exp(min(test_metrics['loss'], 50)):.2f} | "
    f"token accuracy={test_metrics['token_accuracy']:.2%}"
)

hypotheses = []
for start in tqdm(
    range(0, len(test_pairs), CFG["batch_size"]), desc="Greedy dịch test"
):
    batch = test_pairs[start:start + CFG["batch_size"]]
    hypotheses.extend(greedy_decode(
        [src for src, _ in batch], model, src_vocab, tgt_vocab,
        CFG["max_src_tokens"], CFG["max_tgt_tokens"] + 1,
    ))
references = [tokens_to_text(tgt.split()) for _, tgt in test_pairs]
bleu_metric, chrf_metric = BLEU(tokenize="13a"), CHRF()
bleu_score = bleu_metric.corpus_score(hypotheses, [references])
chrf_score = chrf_metric.corpus_score(hypotheses, [references])
print(f"Test: {len(test_pairs):,} cặp sau lọc/lấy mẫu.")
print("BLEU:", bleu_score)
print("chrF:", chrf_score)
print("BLEU signature:", bleu_metric.get_signature())
print("chrF signature:", chrf_metric.get_signature())

evaluation = {
    "checkpoint_epoch": best_checkpoint["epoch"],
    "test_pairs": len(test_pairs),
    "teacher_forced_loss": test_metrics["loss"],
    "teacher_forced_token_accuracy": test_metrics["token_accuracy"],
    "bleu": bleu_score.score,
    "chrf": chrf_score.score,
    "bleu_signature": str(bleu_metric.get_signature()),
    "chrf_signature": str(chrf_metric.get_signature()),
}
write_json(RUN_DIR / "metrics.json", evaluation)
with (RUN_DIR / "test_predictions.tsv").open(
    "w", encoding="utf-8", newline=""
) as handle:
    writer = csv.writer(handle, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["source_en", "reference_vi", "prediction_vi"])
    for (src, _), reference, hypothesis in zip(test_pairs, references, hypotheses):
        writer.writerow([src, reference, hypothesis])

for i in range(min(5, len(test_pairs))):
    print(f"\n[{i + 1}] EN: {tokens_to_text(test_pairs[i][0].split())}")
    print("REF:", references[i])
    print("PRED:", hypotheses[i])


# CELL 21
def translate(text, return_attention=False):
    result = greedy_decode(
        [text], model, src_vocab, tgt_vocab,
        CFG["max_src_tokens"], CFG["max_tgt_tokens"] + 1,
        return_attention=return_attention,
    )
    if return_attention:
        predictions, (generated, source_labels), attention = result
        target_labels = []
        for idx in generated[0]:
            target_labels.append(tgt_vocab.itos[idx])
            if idx == EOS:
                break
        matrix = attention[0, :len(target_labels), :len(source_labels[0])]
        return predictions[0], source_labels[0], target_labels, matrix
    return result[0]

your_sentence = "I am very happy to be here today."
prediction, source_labels, target_labels, attention_matrix = translate(
    your_sentence, return_attention=True
)
print("EN:", your_sentence)
print("VI:", prediction)
print(
    "Token EN ngoài vocabulary:",
    [token for token in tokenize(your_sentence) if token not in src_vocab.stoi],
)

fig, ax = plt.subplots(
    figsize=(max(7, len(source_labels) * 0.45), max(3, len(target_labels) * 0.25))
)
heatmap = ax.imshow(attention_matrix, aspect="auto", cmap="Blues")
ax.set_xticks(range(len(source_labels)), source_labels, rotation=60, ha="right")
ax.set_yticks(range(len(target_labels)), target_labels)
ax.set(xlabel="Token tiếng Anh", ylabel="Token decoder sinh", title="Attention khi dịch")
fig.colorbar(heatmap, ax=ax, label="Trọng số")
plt.tight_layout()
plt.show()


# CELL 23
def load_translator(checkpoint_path, device=None):
    device = DEVICE if device is None else torch.device(device)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    source_vocab = Vocabulary(saved["src_itos"])
    target_vocab = Vocabulary(saved["tgt_itos"])
    restored = make_model(saved["config"], source_vocab, target_vocab)
    restored.load_state_dict(saved["model_state"])
    restored.to(device).eval()
    return restored, source_vocab, target_vocab, saved["config"]

CHECKPOINT_PATH = globals().get("BEST_PATH")
# Kernel mới: bỏ comment dòng tiếp theo và sửa đường dẫn.
# CHECKPOINT_PATH = Path(r"E:\...\outputs\s2s\...\best_model.pt")
if CHECKPOINT_PATH is None:
    raise ValueError("Hãy đặt CHECKPOINT_PATH tới file best_model.pt đã lưu.")
saved_model, saved_src_vocab, saved_tgt_vocab, saved_config = load_translator(
    CHECKPOINT_PATH
)

def translate_saved(text):
    return greedy_decode(
        [text], saved_model, saved_src_vocab, saved_tgt_vocab,
        saved_config["max_src_tokens"], saved_config["max_tgt_tokens"] + 1,
    )[0]

print("Checkpoint:", CHECKPOINT_PATH)
saved_sentence = "I am very happy to be here today."
print("Dịch từ model đã tải:", translate_saved(saved_sentence))
if "model" in globals() and "translate" in globals():
    assert translate_saved(saved_sentence) == translate(saved_sentence)
    print("Checkpoint tải lại cho cùng kết quả.")
