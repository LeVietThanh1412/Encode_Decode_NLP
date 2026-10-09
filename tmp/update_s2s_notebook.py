from pathlib import Path
import ast
import json
import textwrap

path = Path('encode_decode_s2s.ipynb')
notebook = json.loads(path.read_text(encoding='utf-8'))
backup = Path('tmp/encode_decode_s2s.before_kaggle.ipynb')
if not backup.exists():
    backup.write_bytes(path.read_bytes())
assert len(notebook['cells']) == 25, 'Notebook changed; inspect its cells before editing.'


def update(index, content):
    cell = notebook['cells'][index]
    cell['source'] = textwrap.dedent(content).strip().splitlines(keepends=True)
    if cell['source']:
        cell['source'][-1] = cell['source'][-1].rstrip('\n') + '\n'
    if cell['cell_type'] == 'code':
        ast.parse(''.join(cell['source']))
        cell['outputs'] = []
        cell['execution_count'] = None


update(0, '''
# Seq2Seq Anh → Việt: Encoder–Decoder GRU có Attention

Huấn luyện **dịch văn bản Anh → Việt từ đầu** bằng embedding, encoder GRU hai chiều và decoder GRU một chiều có Attention. Mỗi dòng TSV là `câu tiếng Anh<TAB>câu tiếng Việt`, không có header.

**Chạy trên Kaggle:** đặt dự án tại `/kaggle/working/Encode_Decode_NLP`, để ba file `train.tsv`, `validation.tsv`, `test.tsv` trong `/kaggle/working/Encode_Decode_NLP/data`, chọn GPU trong Settings rồi **Run All**. Cell cấu hình đã ưu tiên đúng đường dẫn này. Checkpoint và kết quả nằm trong `/kaggle/working/Encode_Decode_NLP/outputs/s2s/<tên-lần-chạy>/`.

**Mặc định trên Kaggle:** dùng toàn bộ cặp câu hợp lệ sau lọc, tối đa 20 epoch, batch 64, embedding/hidden 256, tự chọn CUDA và dùng mixed precision khi có GPU. Chọn `RUN_PROFILE = "quick"` nếu chỉ muốn thử 5.000 cặp câu. Chạy local mặc định dùng profile `quick`.

**Validation/test hiện tại trùng nhau:** notebook kiểm tra và tạo validation mới từ train theo nhóm câu nguồn; loại nguồn test khỏi train để tránh rò rỉ. Vocabulary chỉ được học từ train đã chọn.

Lộ trình: **cấu hình → dữ liệu song ngữ → vocabulary/DataLoader → encoder/decoder → train/checkpoint → đánh giá → dịch và xem Attention → tải lại model**. Muốn tiếp tục một lần train bị ngắt, đặt `RESUME_CHECKPOINT` ở cell cấu hình tới `last_model.pt`.
''')

update(1, '''
## 1. Thư viện, GPU và cấu hình Kaggle

Kaggle: bật GPU trong **Settings → Accelerator**, đặt ba file TSV ở `/kaggle/working/Encode_Decode_NLP/data`, rồi chạy các cell theo thứ tự. Notebook in thiết bị, đường dẫn dữ liệu, cấu hình và thư mục kết quả trước khi train. Nếu gói còn thiếu, cell dưới cài vào kernel hiện tại; bật Internet khi cần tải gói.

Local: chọn kernel `.venv-s2s` tại `.venv-s2s/Scripts/python.exe`. Nếu tạo môi trường mới, cài `torch numpy matplotlib tqdm sacrebleu ipykernel`. Dùng Python 3.10+ và PyTorch 2.2+.

Sửa `RUN_PROFILE`, `DATA_DIR_OVERRIDE`, `OUTPUT_ROOT_OVERRIDE`, `RESUME_CHECKPOINT` và `CFG` trong cell cấu hình. `CFG["epochs"]` là **tổng số epoch**, kể cả epoch đã chạy trước khi resume.
''')

update(4, '''
# Kaggle: Settings -> Accelerator -> GPU, sau đó Restart Kernel -> Run All.
IS_KAGGLE = Path("/kaggle/working").is_dir()
RUN_PROFILE = "kaggle" if IS_KAGGLE else "quick"  # "kaggle" hoặc "quick"
DATA_DIR_OVERRIDE = (
    Path("/kaggle/working/Encode_Decode_NLP/data") if IS_KAGGLE else None
)
OUTPUT_ROOT_OVERRIDE = None  # Ví dụ: Path("/kaggle/working/my_s2s_runs")
RESUME_CHECKPOINT = None
# Tiếp tục train: thay None bằng đường dẫn last_model.pt đã lưu.
# RESUME_CHECKPOINT = Path("/kaggle/working/Encode_Decode_NLP/outputs/s2s/.../last_model.pt")

if RUN_PROFILE not in {"quick", "kaggle"}:
    raise ValueError('RUN_PROFILE phải là "quick" hoặc "kaggle".')
full_training = RUN_PROFILE == "kaggle"
CFG = {
    "seed": 42,
    "max_train_samples": None if full_training else 5_000,
    "max_validation_samples": None if full_training else 500,
    "max_test_samples": None if full_training else 500,
    "validation_fraction": 0.05,
    "max_src_tokens": 60 if full_training else 50,
    "max_tgt_tokens": 60 if full_training else 50,
    "src_vocab_size": 30_000 if full_training else 12_000,
    "tgt_vocab_size": 30_000 if full_training else 12_000,
    "min_freq": 2,
    "embedding_dim": 256 if full_training else 128,
    "hidden_dim": 256 if full_training else 128,
    "dropout": 0.2,
    "batch_size": 64 if full_training else 32,
    "epochs": 20 if full_training else 5,
    "learning_rate": 1e-3,
    "gradient_clip": 1.0,
    "patience": 5 if full_training else 3,
    "cpu_threads": 4,
    "num_workers": 2 if IS_KAGGLE else 0,
    "use_amp": True,
}

# Resume giữ nguyên model, tokenizer và phép chia dữ liệu của lần train cũ.
resume_checkpoint = None
if RESUME_CHECKPOINT is not None:
    RESUME_CHECKPOINT = Path(RESUME_CHECKPOINT).expanduser()
    if not RESUME_CHECKPOINT.is_file():
        raise FileNotFoundError(RESUME_CHECKPOINT)
    resume_checkpoint = torch.load(
        RESUME_CHECKPOINT, map_location="cpu", weights_only=True
    )
    if "optimizer_state" not in resume_checkpoint:
        raise ValueError("Resume cần last_model.pt; best_model.pt dùng để dịch.")
    runtime_options = {
        key: CFG[key] for key in ("epochs", "num_workers", "cpu_threads", "use_amp")
    }
    CFG = {**resume_checkpoint["config"], **runtime_options}
    if CFG["epochs"] <= resume_checkpoint["epoch"]:
        raise ValueError("Hãy tăng CFG['epochs'] lớn hơn epoch trong checkpoint.")

assert CFG["epochs"] >= 1 and CFG["batch_size"] >= 1
assert 0 < CFG["validation_fraction"] < 1
assert CFG["min_freq"] >= 1 and CFG["patience"] >= 1
assert CFG["num_workers"] >= 0 and CFG["cpu_threads"] >= 1
assert min(CFG["max_src_tokens"], CFG["max_tgt_tokens"]) >= 1
assert min(CFG["src_vocab_size"], CFG["tgt_vocab_size"]) >= 5
for name in ("max_train_samples", "max_validation_samples", "max_test_samples"):
    assert CFG[name] is None or CFG[name] > 0, name


def find_data_dir(override=None):
    """Đường dẫn chỉ định phải có đủ 3 TSV; nếu không, báo lỗi ngay."""
    def complete(directory):
        return all((directory / f"{split}.tsv").is_file()
                   for split in ("train", "validation", "test"))

    if override is not None:
        directory = Path(override).expanduser()
        if not complete(directory):
            raise FileNotFoundError(
                f"Thiếu train.tsv, validation.tsv hoặc test.tsv trong {directory}. "
                "Kiểm tra thư mục dự án trên Kaggle hoặc sửa DATA_DIR_OVERRIDE."
            )
        return directory.resolve()

    candidates = [Path("/kaggle/working/Encode_Decode_NLP/data")]
    for root in (Path.cwd(), *Path.cwd().parents):
        candidates.extend((root / "data", root))
    for directory in candidates:
        if complete(directory):
            return directory.resolve()
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for train_file in sorted(kaggle_input.rglob("train.tsv")):
            if complete(train_file.parent):
                return train_file.parent.resolve()
    raise FileNotFoundError(
        "Không tìm thấy đủ 3 file TSV. Kaggle: đặt chúng trong "
        "/kaggle/working/Encode_Decode_NLP/data hoặc sửa DATA_DIR_OVERRIDE."
    )


DATA_DIR = find_data_dir(DATA_DIR_OVERRIDE)
PROJECT_DIR = DATA_DIR.parent
OUTPUT_ROOT = (
    Path(OUTPUT_ROOT_OVERRIDE) if OUTPUT_ROOT_OVERRIDE is not None
    else Path("/kaggle/working/Encode_Decode_NLP/outputs/s2s") if IS_KAGGLE
    else PROJECT_DIR / "outputs" / "s2s"
)
RUN_DIR = OUTPUT_ROOT / (
    datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
)
RUN_DIR.mkdir(parents=True, exist_ok=True)

set_seed(CFG["seed"])
torch.set_num_threads(CFG["cpu_threads"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = bool(CFG["use_amp"] and DEVICE.type == "cuda")
print("Profile:", RUN_PROFILE, "| PyTorch:", torch.__version__)
print("Thiết bị:", DEVICE, "| Mixed precision:", AMP_ENABLED)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
elif IS_KAGGLE:
    print("Đang dùng CPU. Bật GPU trong Settings để train cấu hình đầy đủ nhanh hơn.")
print("Data:", DATA_DIR)
print("Kết quả:", RUN_DIR)
if RESUME_CHECKPOINT is not None:
    print("Tiếp tục từ:", RESUME_CHECKPOINT)
print(json.dumps(CFG, ensure_ascii=False, indent=2))
''')

update(8, '''
## 3. Vocabulary và DataLoader

`<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3`. Encoder nhận `câu Anh + EOS`. Target là `BOS + câu Việt + EOS`.

Teacher forcing: decoder input = `target[:, :-1]`, nhãn = `target[:, 1:]`. Padding theo độ dài lớn nhất trong batch; bỏ PAD khỏi loss và Attention. Kaggle mặc định dùng 2 worker, local dùng 0; sửa `CFG["num_workers"]` nếu cần. `pin_memory` chỉ bật khi dùng CUDA.
''')
source10 = ''.join(notebook['cells'][10]['source'])
source10 = source10.replace('"num_workers": 0,', '"num_workers": CFG["num_workers"],\n    "persistent_workers": CFG["num_workers"] > 0,')
source10 += '''
if resume_checkpoint is not None:
    if (src_vocab.itos != resume_checkpoint["src_itos"]
            or tgt_vocab.itos != resume_checkpoint["tgt_itos"]):
        raise ValueError("Vocabulary khác checkpoint; kiểm tra dữ liệu/config để resume.")
    if split_summary["file_sha256"] != resume_checkpoint["data_summary"]["file_sha256"]:
        raise ValueError("File dữ liệu đã thay đổi; không tiếp tục checkpoint trên split khác.")
'''
update(10, source10)

source11 = ''.join(notebook['cells'][11]['source'])
source11 += '''
**Ví dụ encoder trong Seq2Seq:** `I love machine learning` → token/ID tiếng Anh → embedding → GRU hai chiều → `memory` theo từng vị trí và `initial_decoder_hidden`. Decoder nhận trạng thái này cùng BOS rồi học sinh `Tôi yêu thích học máy` và EOS. Cell kế tiếp in các tensor encoder của một câu train thật để theo dõi đường dữ liệu.
'''
update(11, source11)
source12 = ''.join(notebook['cells'][12]['source'])
source12 = source12.replace(
    'attention = scores.softmax(dim=-1)',
    '# Softmax FP32 để ổn định khi các phép chiếu dùng mixed precision.\n'
    '        attention = scores.float().softmax(dim=-1).to(memory.dtype)'
)
update(12, source12)
source13 = ''.join(notebook['cells'][13]['source'])
source13 += '''
with torch.no_grad():
    encoder_memory, decoder_start, encoder_mask = model.encoder(
        sample_src.to(DEVICE), sample_lengths
    )
    encoder_embeddings = model.encoder.embedding(sample_src.to(DEVICE))
print("\\nVí dụ encoder Seq2Seq, câu Anh:", train_pairs[0][0])
print("Câu Việt cần học:", train_pairs[0][1])
print("Embedding nguồn:", tuple(encoder_embeddings.shape))
print("Memory encoder:", tuple(encoder_memory.shape))
print("Hidden khởi tạo decoder:", tuple(decoder_start.shape))
print("8 chiều đầu của hidden:", decoder_start[0, 0, :8].cpu().tolist())
'''
update(13, source13)

update(14, '''
## 5. Huấn luyện, checkpoint và tiếp tục train

Cross-entropy được trung bình theo **token đích khác PAD**, gồm EOS. Dùng Adam, gradient clipping, mixed precision khi có CUDA và early stopping theo validation loss. Test chỉ đánh giá sau khi chọn checkpoint.

`best_model.pt` lưu model có validation loss tốt nhất để dịch. `last_model.pt` lưu model/optimizer/scaler ở epoch vừa hoàn tất, vocabulary, cấu hình, trạng thái random và model tốt nhất để **resume ở một phiên Kaggle mới**. Checkpoint được ghi qua file tạm rồi thay thế, tránh để lại file đích đang ghi dở. `history.json` lưu các epoch đã hoàn tất.

Muốn resume: giữ/tải thư mục kết quả, đặt `RESUME_CHECKPOINT` tới `last_model.pt`, tăng tổng `CFG["epochs"]`, rồi **Run All**. Lần resume tạo thư mục kết quả mới. Một epoch đang chạy dở sẽ chạy lại; checkpoint được lưu sau mỗi epoch hoàn tất. Loss và token accuracy dùng teacher forcing, không phải chất lượng bản dịch tự sinh.

API tham khảo: [Automatic Mixed Precision](https://docs.pytorch.org/docs/stable/amp.html), [Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).
''')

update(15, '''
# Hỗ trợ cả torch.amp mới và PyTorch 2.2.
if hasattr(torch.amp, "GradScaler"):
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)
else:
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)


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
        if token_count == 0:
            raise ValueError("Batch không có nhãn đích hợp lệ.")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=DEVICE.type, dtype=torch.float16, enabled=AMP_ENABLED
            ):
                logits, _ = model(src, lengths, decoder_inputs)
            loss_sum = criterion(
                logits.float().reshape(-1, len(tgt_vocab)), labels.reshape(-1)
            )
            loss = loss_sum / token_count
            if not torch.isfinite(loss):
                raise RuntimeError("Loss không hữu hạn; thử giảm learning_rate hoặc tắt use_amp.")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), CFG["gradient_clip"])
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss_sum.detach().item())
        total_tokens += token_count
        correct_tokens += int(
            ((logits.argmax(-1) == labels) & valid_tokens).sum().item()
        )
        progress.set_postfix(loss=f"{total_loss / total_tokens:.3f}")
    if total_tokens == 0:
        raise ValueError("DataLoader rỗng.")
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


def save_checkpoint(payload, path):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def model_state_on_cpu():
    # clone cần thiết: best state phải giữ nguyên khi model tiếp tục cập nhật trên CPU.
    return {key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()}


def random_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": loader_generator.get_state(),
    }


def restore_random_state(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]
    ))
    torch.set_rng_state(state["torch"])
    loader_generator.set_state(state["loader"])
    if torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])


write_json(RUN_DIR / "config.json", CFG)
write_json(RUN_DIR / "data_summary.json", split_summary)
write_json(RUN_DIR / "vocab.json", {"src_itos": src_vocab.itos, "tgt_itos": tgt_vocab.itos})

optimizer = torch.optim.Adam(model.parameters(), lr=CFG["learning_rate"])
BEST_PATH = RUN_DIR / "best_model.pt"
LAST_PATH = RUN_DIR / "last_model.pt"
history = []
best_val_loss, best_epoch, stale_epochs = float("inf"), 0, 0
best_state, first_epoch = None, 1


def inference_checkpoint(state, epoch, val_loss):
    return {
        "model_state": state,
        "config": dict(CFG),
        "src_itos": src_vocab.itos,
        "tgt_itos": tgt_vocab.itos,
        "epoch": epoch,
        "validation_loss": val_loss,
        "data_summary": split_summary,
        "torch_version": str(torch.__version__),
    }


if resume_checkpoint is not None:
    model.load_state_dict(resume_checkpoint["model_state"])
    optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
    if AMP_ENABLED and resume_checkpoint.get("scaler_state"):
        scaler.load_state_dict(resume_checkpoint["scaler_state"])
    history = list(resume_checkpoint["history"])
    best_val_loss = resume_checkpoint["best_validation_loss"]
    best_epoch = resume_checkpoint["best_epoch"]
    stale_epochs = resume_checkpoint["stale_epochs"]
    best_state = resume_checkpoint["best_model_state"]
    first_epoch = resume_checkpoint["epoch"] + 1
    save_checkpoint(inference_checkpoint(best_state, best_epoch, best_val_loss), BEST_PATH)
    write_json(RUN_DIR / "history.json", history)
    restore_random_state(resume_checkpoint["random_state"])
    print(f"Tiếp tục từ epoch {first_epoch}; đã giữ model tốt nhất epoch {best_epoch}.")


for epoch in range(first_epoch, CFG["epochs"] + 1):
    start = time.perf_counter()
    train_metrics = run_epoch(train_loader, optimizer, f"Train {epoch}/{CFG['epochs']}")
    val_metrics = run_epoch(validation_loader, description="Validation")
    elapsed = time.perf_counter() - start
    improved = val_metrics["loss"] < best_val_loss
    if improved:
        best_val_loss, best_epoch, stale_epochs = val_metrics["loss"], epoch, 0
        best_state = model_state_on_cpu()
        save_checkpoint(inference_checkpoint(best_state, epoch, best_val_loss), BEST_PATH)
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
    latest = inference_checkpoint(model_state_on_cpu(), epoch, val_metrics["loss"])
    latest.update({
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "history": history,
        "best_validation_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_model_state": best_state,
        "stale_epochs": stale_epochs,
        "random_state": random_state(),
    })
    save_checkpoint(latest, LAST_PATH)
    print(
        f"Epoch {epoch:02d} | train loss={train_metrics['loss']:.4f} | "
        f"val loss={val_metrics['loss']:.4f} | "
        f"val token acc={val_metrics['token_accuracy']:.2%} | "
        f"{elapsed:.1f}s" + (" | lưu best + last" if improved else " | lưu last")
    )
    if stale_epochs >= CFG["patience"]:
        print("Early stopping.")
        break

print(f"\\nBest epoch: {best_epoch} | validation loss: {best_val_loss:.4f}")
print("Model để dịch:", BEST_PATH)
print("Checkpoint để tiếp tục train:", LAST_PATH)
''')

source18 = ''.join(notebook['cells'][18]['source'])
source18 = source18.replace('    if not texts:',
    '    if max_new_tokens < 1:\n        raise ValueError("max_new_tokens phải >= 1.")\n    if not texts:')
source18 = source18.replace('return ([], [], None) if return_attention else []',
    'return ([], ([], []), None) if return_attention else []')
update(18, source18)
source23 = ''.join(notebook['cells'][23]['source'])
source23 = source23.replace(
    '# CHECKPOINT_PATH = Path(r"E:\\...\\outputs\\s2s\\...\\best_model.pt")',
    '# CHECKPOINT_PATH = Path("/kaggle/working/Encode_Decode_NLP/outputs/s2s/.../best_model.pt")'
)
update(23, source23)
update(24, '''
## 9. Chạy train đầy đủ trên Kaggle

1. Đặt dự án ở `/kaggle/working/Encode_Decode_NLP`, kiểm tra `data/train.tsv`, `data/validation.tsv`, `data/test.tsv`.
2. Bật GPU trong Settings. Giữ `RUN_PROFILE = "kaggle"`, `RESUME_CHECKPOINT = None` khi train mới, rồi **Run All**.
3. Mặc định dùng toàn bộ câu hợp lệ, vocabulary tối đa 30.000 token mỗi ngôn ngữ, embedding/hidden 256, batch 64 và tối đa 20 epoch. Bộ lọc 60 token vẫn loại câu dài; xem thống kê để biết chính xác số cặp được dùng. Giảm batch xuống 32 nếu thiếu bộ nhớ GPU.
4. Tải/giữ `best_model.pt` để dịch và `last_model.pt` để tiếp tục train. Với phiên mới, khôi phục TSV giống trước, đặt `RESUME_CHECKPOINT` tới `last_model.pt`, tăng tổng `CFG["epochs"]`, rồi chạy lại. Notebook khôi phục model, optimizer, scaler, vocabulary và lịch sử; kiểm tra hash dữ liệu trước khi tiếp tục.

Kết quả: `best_model.pt`, `last_model.pt`, `config.json`, `vocab.json`, `data_summary.json`, `history.json`, `metrics.json`, `test_predictions.tsv` trong thư mục được in ở phần 1. Validation loss chọn model; BLEU/chrF chỉ báo trên test sau đó. Profile `quick` phục vụ thử pipeline, không đủ để kết luận chất lượng dịch.

Ví dụ học NLP: encoder nhận token tiếng Anh → embedding → GRU → memory/hidden; decoder nhận BOS/token tiếng Việt trước → GRU + Attention → logits từ vựng tiếng Việt → token tiếp theo. Mô hình và tokenizer được học từ các cặp câu trong TSV.
''')

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        ast.parse(''.join(cell['source']))
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(f'Updated {path}: {len(notebook["cells"])} cells; all code cells parse.')
