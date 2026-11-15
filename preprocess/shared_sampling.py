"""Feed several partition writers from one temporal sampling pass."""


def write_sampling_prefixes(sample_layer, writers):
    """Writers yield (left, right, nodes, times) and receive sampled hop prefixes.

    Each writer owns its maps and cache state. Only the current batch's sample
    is shared; no sampled graph history is retained between batches.
    """
    try:
        requests = [next(writer, None) for _, writer in writers]
        while any(request is not None for request in requests):
            if any(request is None for request in requests):
                raise ValueError('Partition writers ended at different batches')
            left, right, nodes, times = requests[0]
            if any(request[:2] != (left, right) for request in requests[1:]):
                raise ValueError('Partition writers must use the same batch boundaries')
            sampled = sample_layer(nodes, times)
            if any(hops > len(sampled) for hops, _ in writers):
                raise ValueError('Sampler returned fewer hops than requested')
            for i, (hops, writer) in enumerate(writers):
                try:
                    requests[i] = writer.send(sampled[:hops])
                except StopIteration:
                    requests[i] = None
            del sampled
    finally:
        for _, writer in writers:
            writer.close()
