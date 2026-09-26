import time
import jax


def compile_and_run(fn, *args, label="training"):
    """Compile and execute a JIT function with timing."""
    print(f"JAX Backend: {jax.default_backend()} | Devices: {jax.devices()}")
    print(f"Compiling / Warming up JAX graph for {label}...")
    t0 = time.perf_counter()
    compiled_fn = jax.jit(fn).lower(*args).compile()
    warmup_time = time.perf_counter() - t0
    print(f"WARMUP / COMPILATION TIME: {warmup_time:.4f} seconds")

    print(f"Running {label}...")
    t0 = time.perf_counter()
    results = compiled_fn(*args)
    jax.tree.map(lambda x: x.block_until_ready(), results)
    exec_time = time.perf_counter() - t0
    print(f"WARM EXECUTION TIME: {exec_time:.4f} seconds")
    print(f"TOTAL TIME (Warmup + Execution): {warmup_time + exec_time:.4f} seconds")
    return results, warmup_time, exec_time


def run_multi_seeds(single_run, seed_keys, mode="vmap", chunk_size=None):
    """Execute single_run across seed_keys using parallel vmap, chunked vmap, or sequential execution."""
    num_seeds = len(seed_keys)
    if mode == "vmap":
        if chunk_size is not None and chunk_size < num_seeds:
            import jax.numpy as jnp
            num_chunks = (num_seeds + chunk_size - 1) // chunk_size
            print(f"JAX Backend: {jax.default_backend()} | Devices: {jax.devices()}")
            print(f"Running chunked vmap: {num_seeds} seeds in {num_chunks} chunks of size {chunk_size}...")

            vmap_train = jax.vmap(single_run)
            first_chunk_keys = seed_keys[:chunk_size]
            t0 = time.perf_counter()
            compiled_vmap_train = jax.jit(vmap_train).lower(first_chunk_keys).compile()
            warmup_time = time.perf_counter() - t0
            print(f"WARMUP / COMPILATION TIME: {warmup_time:.4f} seconds")

            t0 = time.perf_counter()
            all_losses_l, all_mean_returns_l, all_Y_means_l = [], [], []
            for c_idx in range(num_chunks):
                c_start = c_idx * chunk_size
                c_end = min(c_start + chunk_size, num_seeds)
                chunk_k = seed_keys[c_start:c_end]
                if len(chunk_k) == chunk_size:
                    l, r, y, p = compiled_vmap_train(chunk_k)
                else:
                    l, r, y, p = jax.jit(vmap_train)(chunk_k)
                jax.tree.map(lambda x: x.block_until_ready(), (l, r, y))
                all_losses_l.append(l)
                all_mean_returns_l.append(r)
                all_Y_means_l.append(y)
                print(f"  Completed Chunk {c_idx + 1}/{num_chunks} (Seeds {c_start}..{c_end-1})")
            exec_time = time.perf_counter() - t0

            all_losses = jnp.concatenate(all_losses_l, axis=0)
            all_mean_returns = jnp.concatenate(all_mean_returns_l, axis=0)
            all_Y_means = jnp.concatenate(all_Y_means_l, axis=0)
        else:
            results, warmup_time, exec_time = compile_and_run(
                jax.vmap(single_run), seed_keys, label=f"parallel vmap ({num_seeds} seeds)"
            )
            all_losses, all_mean_returns, all_Y_means, _ = results
    else:
        single_train = jax.jit(single_run)
        print(f"JAX Backend: {jax.default_backend()} | Devices: {jax.devices()}")
        print("Compiling / Warming up single-seed JAX graph...")
        t0 = time.perf_counter()
        compiled_single_train = single_train.lower(seed_keys[0]).compile()
        warmup_time = time.perf_counter() - t0
        print(f"WARMUP / COMPILATION TIME: {warmup_time:.4f} seconds")

        print(f"Executing {num_seeds} runs sequentially...")
        t0 = time.perf_counter()
        all_losses_l, all_mean_returns_l, all_Y_means_l = [], [], []
        for s_idx in range(num_seeds):
            l, r, y, p = compiled_single_train(seed_keys[s_idx])
            jax.tree.map(lambda x: x.block_until_ready(), (l, r, y))
            all_losses_l.append(l)
            all_mean_returns_l.append(r)
            all_Y_means_l.append(y)
        exec_time = time.perf_counter() - t0

        import jax.numpy as jnp
        all_losses = jnp.stack(all_losses_l, axis=0)
        all_mean_returns = jnp.stack(all_mean_returns_l, axis=0)
        all_Y_means = jnp.stack(all_Y_means_l, axis=0)
        _ = None

    return all_losses, all_mean_returns, all_Y_means, (warmup_time, exec_time)
