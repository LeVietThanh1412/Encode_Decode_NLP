from pathlib import Path
import ast
import json

path = Path('encode_decode_s2s.ipynb')
notebook = json.loads(path.read_text(encoding='utf-8'))
cell = notebook['cells'][12]
source = ''.join(cell['source'])
old_encoder = '''        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, hidden = self.gru(packed)
        output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=src.size(1)
        )'''
new_encoder = '''        # GRU + packed sequence chạy FP32; các phép chiếu vẫn dùng AMP ở ngoài.
        with torch.autocast(device_type=src.device.type, enabled=False):
            embedded = self.dropout(self.embedding(src))
            packed = pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, hidden = self.gru(packed)
            output, _ = pad_packed_sequence(
                packed_output, batch_first=True, total_length=src.size(1)
            )'''
old_decoder = '''        embedded = self.dropout(self.embedding(input_ids))
        states, hidden = self.gru(embedded, hidden)'''
new_decoder = '''        # Hidden bridge có thể là FP16 dưới AMP; GRU nhận input/hidden cùng FP32.
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            embedded = self.dropout(self.embedding(input_ids))
            states, hidden = self.gru(embedded, hidden.float())'''
assert source.count(old_encoder) == 1 and source.count(old_decoder) == 1
source = source.replace(old_encoder, new_encoder).replace(old_decoder, new_decoder)
ast.parse(source)
cell['source'] = source.splitlines(keepends=True)
cell['execution_count'] = None
cell['outputs'] = []
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print('GRU uses FP32; projections and vocabulary logits support AMP.')
