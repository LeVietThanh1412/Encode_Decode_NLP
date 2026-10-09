import ast
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import nbformat
import torch

torch.set_num_threads(2)
root = Path.cwd()
notebook = nbformat.read(root / 'encode_decode_s2s.ipynb', as_version=4)
nbformat.validate(notebook)
for cell in notebook.cells:
    if cell.cell_type == 'code': ast.parse(cell.source)
config = notebook.cells[4].source
assert 'find_data_dir' not in ''.join(c.source for c in notebook.cells)
assert 'Path("/kaggle/working/Encode_Decode_NLP/data")' in config
assert 'torch.device("cuda:0")' in config
assert 'assert torch.cuda.is_available()' in config
assert all(c.get('execution_count') is None and not c.get('outputs')
           for c in notebook.cells if c.cell_type == 'code')

test_dir = root / 'tmp' / 's2s_clean_smoke'
test_dir.mkdir(parents=True, exist_ok=True)
limits = dict(max_train_samples=128, max_validation_samples=24, max_test_samples=24,
              max_src_tokens=32, max_tgt_tokens=32, src_vocab_size=2000,
              tgt_vocab_size=2000, min_freq=1, embedding_dim=16, hidden_dim=16,
              batch_size=16, dropout=0.2, patience=10, num_workers=0, use_amp=False)


def execute_run(name, epochs, resume_path=None, evaluate=False):
    ns = {'__name__': '__main__'}
    exec(notebook.cells[3].source, ns)
    source = config
    source = source.replace('Path("/kaggle/working/Encode_Decode_NLP/data")', repr(root / 'data'))
    source = source.replace('Path("/kaggle/working/Encode_Decode_NLP/outputs/s2s")', repr(test_dir / name))
    # Path repr is platform-specific; expose that class only inside this CPU harness.
    ns[type(root).__name__] = type(root)
    source = source.replace('RESUME_PATH = None', 'RESUME_PATH = ' + repr(resume_path))
    source = source.replace('assert torch.cuda.is_available(), "Bật GPU trong Kaggle Settings → Accelerator rồi chạy lại."',
                            'CFG.update(' + repr({**limits, 'epochs':epochs}) + ')')
    source = source.replace('torch.device("cuda:0")', 'torch.device("cpu")')
    source = source.replace('torch.cuda.get_device_name(0)', '"CPU smoke test; GPU not available locally"')
    exec(source, ns)
    for i in [6,7,9,10,12,13,15]:
        exec(notebook.cells[i].source, ns)
    last = torch.load(ns['LAST_PATH'], map_location='cpu', weights_only=True)
    if evaluate:
        for i in [16,18,19,21]: exec(notebook.cells[i].source, ns)
        # Exercise the actual load/translate code with an explicit CPU device in this test.
        source = notebook.cells[23].source.replace('load_translator(CHECKPOINT_PATH)',
                                                 'load_translator(CHECKPOINT_PATH, "cpu")')
        exec(source, ns)
        assert ns['translate_saved']('I am very happy to be here today.') == ns['translate']('I am very happy to be here today.')
        metrics = json.loads((ns['RUN_DIR'] / 'metrics.json').read_text(encoding='utf-8'))
        assert all(math.isfinite(metrics[k]) for k in ['teacher_forced_loss','bleu','chrf'])
        assert (ns['RUN_DIR'] / 'test_predictions.tsv').is_file()
        model = ns['model'].eval()
        src, lengths, tgt = ns['sample_src'], ns['sample_lengths'], ns['sample_tgt']
        with torch.no_grad():
            original, attention = model(src, lengths, tgt[:, :-1])
            extra_pad = torch.nn.functional.pad(src, (0,3), value=ns['PAD'])
            padded, padded_attention = model(extra_pad, lengths, tgt[:, :-1])
            assert torch.allclose(original, padded, atol=1e-6)
            assert padded_attention[:, :, -3:].eq(0).all()
            altered = tgt[:, :-1].clone()
            altered[:, -1] = ns['UNK']
            changed, _ = model(src, lengths, altered)
            assert torch.allclose(original[:, :-1], changed[:, :-1], atol=1e-6)
        model.zero_grad(set_to_none=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            logits, attention = model(src, lengths, tgt[:, :-1])
            labels = tgt[:,1:]
            loss = ns['criterion'](logits.float().reshape(-1,len(ns['tgt_vocab'])), labels.reshape(-1)) / labels.ne(ns['PAD']).sum()
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        model.zero_grad(set_to_none=True)
        # Verify every example is used once, batches change between epochs, and padding drops.
        sampler = ns['train_sampler']
        sampler.set_epoch(1)
        batches = list(sampler)
        assert sorted(i for batch in batches for i in batch) == list(range(len(ns['train_dataset'])))
        assert all(1 <= len(batch) <= ns['CFG']['batch_size'] for batch in batches)
        sampler.set_epoch(2)
        assert list(sampler) != batches
        dataset = ns['train_dataset']
        def padded_slots(groups):
            return sum((max(len(dataset[i][0]) for i in group) + max(len(dataset[i][1]) for i in group)) * len(group) for group in groups)
        indices = list(range(len(dataset)))
        ns['random'].Random(42).shuffle(indices)
        random_batches = [indices[i:i+16] for i in range(0,len(indices),16)]
        padding_comparison = {'random_batch_slots':padded_slots(random_batches), 'bucket_batch_slots':padded_slots(batches)}
        assert padding_comparison['bucket_batch_slots'] < padding_comparison['random_batch_slots']
        train_sources = {s for s,_ in ns['train_pairs']}
        val_sources = {s for s,_ in ns['validation_pairs']}
        test_sources = {s for s,_ in ns['test_pairs']}
        assert train_sources.isdisjoint(val_sources | test_sources) and val_sources.isdisjoint(test_sources)
        assert not list(ns['RUN_DIR'].glob('*.tmp'))
        ns['padding_comparison'] = padding_comparison
    return ns, last


continuous, continuous_last = execute_run('continuous', 2, evaluate=True)
partial, partial_last = execute_run('partial', 1)
resumed, resumed_last = execute_run('resumed', 2, partial['LAST_PATH'])
max_diff = max((continuous_last['model_state'][k]-resumed_last['model_state'][k]).abs().max().item()
               for k in continuous_last['model_state'])
assert max_diff == 0.0, max_diff
assert [h['epoch'] for h in resumed['history']] == [1,2]
assert resumed['history'][1]['train_loss'] == continuous['history'][1]['train_loss']
summary = {
    'notebook_schema_and_syntax':'passed', 'fixed_kaggle_data_path':'passed',
    'gpu_required_no_cpu_fallback':True, 'cuda_tested':False,
    'training_samples':128, 'validation_samples':24, 'test_samples':24,
    'continuous_training_loss':[r['train_loss'] for r in continuous['history']],
    'resume_max_weight_difference':max_diff,
    'padding_and_causal_mask':'passed', 'mixed_dtype_forward_backward_cpu_bfloat16':'passed',
    'checkpoint_reload_bleu_chrf_predictions_export':'passed',
    'length_bucketing':continuous['padding_comparison'],
    'run_directory':str(continuous['RUN_DIR']),
}
(test_dir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print('\nFINAL SMOKE TEST SUMMARY')
print(json.dumps(summary,ensure_ascii=False,indent=2))
