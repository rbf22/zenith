# Zenith agent notes

Looped Mamba-3 language model (Dory-style recurrent block) trained with MobileLLM-R1's recipes.

## Environment

- Target hardware: Apple M2, 24 GB, MPS. No CUDA here. The official Mamba-3 kernels (Triton/TileLang) are CUDA-only, so `zenith/mamba3.py` is a pure-PyTorch chunked SSD scan.
- Venv: `.venv` (Python 3.11, created with `uv`). Run tools via `.venv/bin/python`.

## Mojo prototype (`mojo/`)

- Installed in `.venv`: `mojo==1.2.0` and `max==26.7.0` (pinned; both published more than 7 days ago). Run with `.venv/bin/mojo run mojo/ssd_fwd.mojo`.
- GPU imports: `from max.gpu.host import DeviceContext`, `from max.gpu import block_idx, thread_idx`, `from std.memory import stack_allocation`, `from std.time import perf_counter_ns`. In Mojo 1.2 there is no `std.gpu` and no `InlineArray` import.
- `ssd_fwd.mojo`: one-thread-per-state-row forward SSD scan on the M2 GPU. It agrees with the CPU reference to 3.8e-6. Check against PyTorch with `.venv/bin/python mojo/check_ssd.py`.
- Measured at B=4, T=512, H=20, P=64, N=128 (forward only): Mojo GPU kernel ~12 to 13 ms per call; PyTorch MPS chunked `ssd` ~20 ms (`mojo/bench_torch.py`).
- Tried staging q/k/kp in threadgroup memory: no speedup. Suspected cause is the 128-entry per-thread state array living in local memory. Next step is one thread per state entry with warp reductions.
- Shared memory in Mojo 1.2: `from layout import stack_allocation` with `address_space=.SHARED` and a `row_major[...]()` layout declared at module level (`comptime LAY = row_major[3, 64, 32]()`). `std.memory.stack_allocation` with `AddressSpace.THREADGROUP` fails in the GPU offload pass. `barrier` is from `max.gpu`.
- `max.gpu.primitives.block.sum` did not resolve for me: the `block_size=` keyword and positional forms were both rejected. Use a shared scratch tile and a serial reduction on thread 0 instead.
- Calling Mojo from Python (`mojo.importer` + `PythonModuleBuilder`) works. Passing a torch MPS tensor needs `wrap_host_memory`, which requires 16 KB page-aligned pointers. Torch allocations are not aligned, so this path is blocked unless we carve aligned views out of a large buffer.
- `mojo/ssd_chunk_fwd.mojo`: chunked forward (intra-chunk dots plus a carried chunk state). It matches the CPU reference (8.8e-6 at the target shape). Speed at B=4, T=512, H=20 is 207 ms, against 12 ms for the one-thread-per-row kernel. Each thread reads q/k/kp/x rows from global memory with a stride of H*N, so the loads don't coalesce. It needs threadgroup tiling of the Q K^T and M X products to be competitive.
- `mojo/ssd_bwd.mojo`: backward kernel (adjoint recurrence, shared-memory tile reduction). It matches the CPU reference: dx, dxp, dq, dk, dkp within 6e-5 absolute, dadt/dgamma/dbeta within 0.02 absolute (fp32 sums over 64x128).
- Forward+backward at the target shape (B=4, T=512, H=20, P=64, N=128): Mojo ~2400 ms per pair vs PyTorch MPS ~580 ms (`mojo/bench_torch_bwd.py`). Both measured while a training run was using the GPU. The Mojo backward is about 4x slower, so it is not a speedup yet. Likely causes: only B*H=80 blocks of 64 threads, a serial per-chunk loop with a barrier per state tile, and the serial reduction on thread 0.

## Verify

- `.venv/bin/python -m pytest -q --cov=zenith --cov-report=term-missing`: all unit tests and line coverage. Every new module or function needs tests. The only expected miss is the `__main__` guard in `train.py`, which is covered by the subprocess test instead.
- Coverage of the chunked scan vs naive recurrence, causality, TBPTL graph cut, parameter budget, loop sampler, tokenizer-backed data loading, and the CLI end to end.

## Train (local, MPS)

```
.venv/bin/python -m zenith.train \
  --config configs/target/config.json \
  --tokenizer <tokenizer dir> \
  --data "<glob of JSONL with a 'text' key>" \
  --output_dir <out> --seq_len 512 --batch_size 2 --max_steps N --warmup_steps W --compile_scan
```

- `--compile_scan` wraps the SSD scan in `torch.compile`. It matches eager bf16 accuracy against an fp32 reference, and it roughly halves step time on MPS.
- Profiling: `scripts/profile_step.py` (timing, `--compile_scan` to A/B), `scripts/profile_ops.py` (per-operator table), `scripts/bench_ssd.py` (scan variants). Under load the timings are noisy; close background apps first.
- The scan is most of a training step (forward+backward, not just forward). Chunk size 64 is the fastest of 32/64/128/256 for the batched scan. Chunk size does not change the model, only speed.

- Hyperparameters follow MobileLLM-R1 `run_pretrain.sh`: lr 5e-4, AdamW betas (0.9, 0.95), eps 1e-8, weight decay 0.1, cosine to 10%.
- Loop count per step is sampled uniformly from {1..max_loops} (Dory), with a 10% rank-level override to 1 when `world_size > 1`.
- Measured on M2 (fwd+bwd, batch of 4 x 512 tokens): one 640-dim Mamba layer ~155 ms with bf16 autocast (300 ms fp32); 32k LM head + CE ~230 ms; 128k LM head + CE ~3.2 s in fp32. A step runs 8 + 16*k layer passes, so cost scales with the sampled loop count.

## Conventions

- Keep sequence lengths a multiple of `chunk_size` (64 for the target config).
- The LM head is tied to the embedding (`_tied_weights_keys` is a dict in transformers 4.57+).
- Layers in the recurrent block are shared across loops. Only the final loop is tracked by autograd (truncated BPTT depth 1).
