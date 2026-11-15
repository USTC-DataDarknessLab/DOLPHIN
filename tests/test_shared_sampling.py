import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import feat_buffer
from preprocess.shared_sampling import write_sampling_prefixes
from sampler.sampler_gpu import Sampler_GPU


class SharedSamplingTests(unittest.TestCase):
    def test_writer_boundaries_must_match_and_writers_close(self):
        closed = []

        def writer(right):
            try:
                yield (0, right, None, None)
            finally:
                closed.append(right)

        with self.assertRaisesRegex(ValueError, 'boundaries'):
            write_sampling_prefixes(self.fail, [(1, writer(2)), (2, writer(3))])
        self.assertEqual(sorted(closed), [2, 3])

    def test_sampler_failure_closes_both_writers(self):
        closed = []

        def writer(hops):
            try:
                yield (0, 2, None, None)
            finally:
                closed.append(hops)

        def fail(*_):
            raise RuntimeError('sampling failed')

        with self.assertRaisesRegex(RuntimeError, 'sampling failed'):
            write_sampling_prefixes(fail, [(1, writer(1)), (2, writer(2))])
        self.assertEqual(closed, [1, 2])

    def test_two_outputs_match_separate_preprocessing(self):
        # Run the real partition/cache writer and real layer traversal on CPU.
        # Only the CUDA sampling kernel and CUDA transfers are substituted.
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(torch.Tensor, 'cuda', lambda tensor, *a, **k: tensor))
            stack.enter_context(patch.object(torch.cuda, 'empty_cache', lambda: None))
            def find_same_index(first, second, first_index, second_index):
                for i, value in enumerate(first):
                    matches = torch.nonzero(second == value).reshape(-1)
                    if matches.numel():
                        j = matches[0].item()
                        first_index[i], second_index[j] = j, i

            stack.enter_context(patch.object(
                feat_buffer.dgl, 'findSameIndex', find_same_index, create=True,
            ))
            for name in ('zeros', 'empty', 'tensor'):
                original = getattr(torch, name)

                def on_cpu(*args, _original=original, **kwargs):
                    if str(kwargs.get('device', '')).startswith('cuda'):
                        kwargs['device'] = 'cpu'
                    return _original(*args, **kwargs)

                stack.enter_context(patch.object(torch, name, on_cpu))
            stack.enter_context(patch.object(feat_buffer, 'flush_saveBin_conf', lambda: None))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

            def run(label, fanouts, paired):
                path = Path(directory) / label
                path.mkdir()
                np.arange(128, dtype=np.float32).tofile(path / 'node_features.bin')
                np.arange(192, dtype=np.float32).tofile(path / 'edge_features.bin')
                saved = {}

                def save(tensor, filename, addSave=False, **_):
                    key = str(Path(filename).relative_to(path)).replace('.pt', '.bin')
                    value = tensor.detach().clone()
                    saved[key] = torch.cat((saved[key], value)) if addSave and key in saved else value

                sampler = Sampler_GPU.__new__(Sampler_GPU)
                sampler.layer, sampler.fan_nums, sampler.emb_buffer = len(fanouts), fanouts, None
                calls = []

                def sample(nodes, times, mask, **kwargs):
                    # Small deterministic temporal neighborhood (degree < 10).
                    dst = torch.arange(nodes.numel()).repeat_interleave(2)
                    src = (nodes[dst] + torch.tensor([1, 3]).repeat(nodes.numel())) % 64
                    ts = times[dst] - 1
                    return [src, dst, ts, src, nodes, times, times[dst]]

                sampler.sample = sample

                def sample_layer(nodes, times):
                    calls.append(nodes.clone())
                    return Sampler_GPU.sample_layer(sampler, nodes, times)

                sampler.sample_layer = sample_layer
                buffer = feat_buffer.Feat_buffer.__new__(feat_buffer.Feat_buffer)
                buffer.d, buffer.path, buffer.df = 'LASTFM', str(path), None
                buffer.batch_size = 2
                buffer.sampler = sampler
                buffer.node_feat_dim, buffer.edge_feat_dim, buffer.has_ef = 2, 3, True
                buffer.total_node_num = buffer.total_edge_num = 64
                buffer.datas = {
                    'src': torch.tensor([0, 3, 6, 9, 12]),
                    'dst': torch.tensor([1, 4, 7, 10, 13]),
                    'time': torch.arange(5, dtype=torch.float32) + 10,
                    'eid': torch.arange(5),
                }
                with patch.object(feat_buffer, 'saveBin', save):
                    buffer.gen_part_stream(budget=1, cache_budget=48, write_single_hop=paired)
                for file in path.glob('part-*/bucket_config.json'):
                    saved[str(file.relative_to(path))] = json.loads(file.read_text())
                return saved, len(calls)

            paired, paired_calls = run('paired', [10, 10], True)
            single, single_calls = run('single', [10], False)
            double, double_calls = run('double', [10, 10], False)
            self.assertEqual(paired_calls, 3)  # Includes the partial final batch.
            self.assertEqual(single_calls + double_calls, 6)
            expected = {**single, **double}
            self.assertEqual(paired.keys(), expected.keys())
            for key, value in expected.items():
                if isinstance(value, torch.Tensor):
                    self.assertTrue(torch.equal(paired[key], value), key)
                else:
                    self.assertEqual(paired[key], value, key)
            self.assertFalse(torch.equal(
                paired['part-2-[10]/part0_node_map.bin'],
                paired['part-2-[10, 10]/part0_node_map.bin'],
            ))


if __name__ == '__main__':
    unittest.main()
