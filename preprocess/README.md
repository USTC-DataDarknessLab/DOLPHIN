# Write single-hop and multi-hop features in one preprocessing run

The streaming disk preprocessor samples each substream once with the configured
multi-hop fanouts. For `[10, 10]`, it writes both `part-60000-[10, 10]` and
`part-60000-[10]`. The single-hop writer receives only the first sampling result;
the two-hop writer receives both. Each writer produces its own features, maps,
incremental masks and cache configuration. Single-hop training therefore loads
the single-hop partition, rather than the larger two-hop partition.

Using the project's existing `/DOLPHIN` data layout, run:

```bash
python /DOLPHIN/preprocess/disk_preprocess.py --data LASTFM --config /DOLPHIN/config/TGN-2.yml
```

The same behavior applies to `TGAT-2.yml` and `TimeSGN-2.yml`. It uses fixed
substream boundaries (`bucket_optimal=False`, the default). The graph and input
interaction tensors are shared; the writers keep separate cache planning state.
Writing two outputs saves repeated graph sampling but uses additional memory,
feature reads and disk space. Add `--no_single_hop` to produce only the configured
hop count. A one-hop config still produces just one output.

Prepare the other required artifacts once per dataset:

```bash
python /DOLPHIN/preprocess/memory_preprocess.py --data LASTFM
python /DOLPHIN/preprocess/node_memory.py --data LASTFM
python /DOLPHIN/eid_analyze/eid_analyze_parallel.py --data LASTFM --config /DOLPHIN/config/TGN-2.yml
```

The EID analyzer computes the data-dependent schedule once and writes it to both
`eid/LASTFM/1` and `eid/LASTFM/2`. Node-memory preprocessing is independent of the
hop count. `memory_preprocess.py` prepares the edge-expiration metadata for the
single-hop `mem_w_valid` configuration; the two-hop disk configuration does not
use this single-hop eviction map. All examples use the default substream size of
60000.

After preparation, train either model repeatedly using its own config:

```bash
python /DOLPHIN/train.py --data LASTFM --eid_load final --config /DOLPHIN/config/TGN-1.yml --train_conf mem_w_valid
python /DOLPHIN/train.py --data LASTFM --eid_load final --config /DOLPHIN/config/TGN-2.yml --train_conf basic_conf_disk
```

Keep the dataset, features, sampling settings and substream size consistent with
the preprocessed artifacts. No separate single-hop disk preprocessing is needed.

The CPU regression test compares both writers' outputs against independent
single-hop and two-hop runs, including a partial final substream. CUDA kernels
are substituted in this test; full training still requires the custom DGL build.

```bash
python -m unittest discover -s tests -v
```
