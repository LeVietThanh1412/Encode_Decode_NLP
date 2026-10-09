import ast
import contextlib
import json
import math
import os
from pathlib import Path
from unittest.mock import patch

os.environ['MPLBACKEND'] = 'Agg'
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
notebook = json.loads((ROOT / 'encode_decode_s2s.ipynb').read_text(encoding='utf-8'))
OUTPUT = ROOT / 'tmp' / 's2s_smoke'
OUTPUT.mkdir(parents=True, exist_ok=True)


class QuietProgress:
    def __init__(self, iterable, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, **kwargs):
        pass


def run_cell(namespace, index):
    code = ''.join(notebook['cells'][index]['source'])
    exec(compile(code, f'notebook_cell_{index}', 'exec'), namespace)


def fresh_run(epochs, checkpoint=None):
    namespace = {'__name__': '__main__'}
    run_cell(namespace, 3)
    namespace['tqdm'] = QuietProgress
    config_ast = ast.parse(''.join(notebook['cells'][4]['source']))
    for statement in config_ast.body:
        exec(compile(ast.Module(body=[statement], type_ignores=[]), 'config_cell', 'exec'), namespace)
        if isinstance(statement, ast.Assign):
            targets = [target.id for target in statement.targets if isinstance(target, ast.Name)]
            if 'RESUME_CHECKPOINT' in targets:
                namespace['RESUME_CHECKPOINT'] = checkpoint
            if 'DATA_DIR_OVERRIDE' in targets:
                namespace['DATA_DIR_OVERRIDE'] = ROOT / 'data'
            if 'OUTPUT_ROOT_OVERRIDE' in targets:
                namespace['OUTPUT_ROOT_OVERRIDE'] = OUTPUT
            if 'CFG' in targets:
                namespace['CFG'].update({
                    'max_train_samples': 128,
                    'max_validation_samples': 24,
                    'max_test_samples': 24,
                    'src_vocab_size': 2_000,
                    'tgt_vocab_size': 2_000,
                    'min_freq': 1,
                    'embedding_dim': 16,
                    'hidden_dim': 16,
                    'batch_size': 16,
                    'epochs': epochs,
                    'patience': 10,
                    'num_workers': 0,
                    'cpu_threads': 2,
                })
    for index in (6, 7, 9, 10, 12, 13, 15):
        run_cell(namespace, index)
    return namespace


continuous = fresh_run(2)
partial = fresh_run(1)
resumed = fresh_run(2, partial['LAST_PATH'])
torch = resumed['torch']

uninterrupted = torch.load(continuous['LAST_PATH'], map_location='cpu', weights_only=True)
continued = torch.load(resumed['LAST_PATH'], map_location='cpu', weights_only=True)
maximum_difference = max(
    (value - continued['model_state'][key]).abs().max().item()
    for key, value in uninterrupted['model_state'].items()
)
assert maximum_difference == 0.0, f'Resumed weights differ: {maximum_difference}'
assert [row['epoch'] for row in resumed['history']] == [1, 2]
assert resumed['BEST_PATH'].is_file() and resumed['LAST_PATH'].is_file()
assert not list(resumed['RUN_DIR'].glob('*.tmp'))

# Kiểm tra profile Kaggle và đường dẫn mà không tạo thư mục ngoài workspace.
profile_namespace = {'Path': Path, 'torch': torch}
for statement in ast.parse(''.join(notebook['cells'][4]['source'])).body:
    exec(compile(ast.Module(body=[statement], type_ignores=[]), 'kaggle_config', 'exec'), profile_namespace)
    if isinstance(statement, ast.Assign):
        targets = [target.id for target in statement.targets if isinstance(target, ast.Name)]
        if 'IS_KAGGLE' in targets:
            profile_namespace['IS_KAGGLE'] = True
        if 'CFG' in targets:
            break
assert profile_namespace['RUN_PROFILE'] == 'kaggle'
assert profile_namespace['CFG']['max_train_samples'] is None
assert profile_namespace['CFG']['epochs'] == 20
assert profile_namespace['CFG']['hidden_dim'] == 256
kaggle_directory = Path('/kaggle/working/Encode_Decode_NLP/data')
assert profile_namespace['DATA_DIR_OVERRIDE'].as_posix() == kaggle_directory.as_posix()
expected_files = {str(kaggle_directory / f'{split}.tsv') for split in ('train', 'validation', 'test')}
with patch.object(Path, 'is_file', lambda path: str(path) in expected_files), \
        patch.object(Path, 'resolve', lambda path: path):
    assert resumed['find_data_dir'](kaggle_directory) == kaggle_directory
try:
    resumed['find_data_dir'](OUTPUT / 'missing_data')
except FileNotFoundError:
    pass
else:
    raise AssertionError('Explicit missing directory silently fell back to local data.')

# Kiểm tra PAD không làm thay đổi logits ở các vị trí thật.
model = resumed['model'].eval()
src, lengths, target = resumed['sample_src'], resumed['sample_lengths'], resumed['sample_tgt']
with torch.no_grad():
    original_logits, _ = model(src, lengths, target[:, :-1])
    padded_logits, padded_attention = model(
        torch.cat((src, torch.zeros(src.size(0), 3, dtype=torch.long)), dim=1),
        lengths, target[:, :-1]
    )
    assert torch.allclose(original_logits, padded_logits, atol=1e-6)
    assert torch.all(padded_attention[:, :, -3:] == 0)
    changed_inputs = target[:, :-1].clone()
    changed_inputs[:, -1] = resumed['UNK']
    changed_logits, _ = model(src, lengths, changed_inputs)
    assert torch.allclose(original_logits[:, :-1], changed_logits[:, :-1], atol=1e-6)

for index in (16, 18, 19, 21, 23):
    run_cell(resumed, index)
metrics = json.loads((resumed['RUN_DIR'] / 'metrics.json').read_text(encoding='utf-8'))
assert metrics['test_pairs'] == 24 and math.isfinite(metrics['teacher_forced_loss'])
assert (resumed['RUN_DIR'] / 'test_predictions.tsv').is_file()
assert resumed['greedy_decode']([], model, resumed['src_vocab'], resumed['tgt_vocab'], 50, 10) == []
assert resumed['greedy_decode'](
    [], model, resumed['src_vocab'], resumed['tgt_vocab'], 50, 10, return_attention=True
) == ([], ([], []), None)
try:
    resumed['greedy_decode'](['hello'], model, resumed['src_vocab'], resumed['tgt_vocab'], 50, 0)
except ValueError:
    pass
else:
    raise AssertionError('Zero generation limit was not rejected.')

summary = {
    'training_epochs_checked': [2, 1, 'resume_to_2'],
    'samples': resumed['split_summary']['selected_pairs'],
    'training_loss': [row['train_loss'] for row in resumed['history']],
    'resume_max_weight_difference': maximum_difference,
    'kaggle_path_and_profile': 'passed',
    'padding_and_causal_decoder': 'passed',
    'inference_checkpoint_round_trip': 'passed',
    'bleu_chrf_and_predictions_export': 'passed',
    'cuda_tested': torch.cuda.is_available(),
    'run_directory': str(resumed['RUN_DIR']),
}
(OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print('\nSMOKE TEST PASSED\n' + json.dumps(summary, indent=2))
