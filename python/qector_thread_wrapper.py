# PROPRIETARY AND CONFIDENTIAL
# Copyright (c) 2026 Guillaume Lessard / qector-decoder-v3
# All Rights Reserved. NDA evaluation only. Do not redistribute.
"""
QECTOR IonQ v1.7.7 Dynamic Thread Allocation Wrapper
=====================================================
Automatically sets RAYON_NUM_THREADS based on the workload size
(qubit count / code distance) to maximize throughput.

Rules:
    qubits < 100   ->  1 thread  (avoids contention overhead)
    100 <= qubits < 500  ->  2 threads
    500 <= qubits < 2000 ->  4 threads  (requires 4+ physical cores)
    qubits >= 2000       ->  min(8, physical_cores)

Usage:
    from qector_thread_wrapper import configure_threads, decode_with_optimal_threads
    configure_threads(n_qubits=70)   # sets env before any Rayon pool init
"""

import os
import multiprocessing


def physical_core_count():
    """Return the number of physical CPU cores (not hyperthreads)."""
    try:
        count = int(os.cpu_count() or 1)
        # On Windows, os.cpu_count() returns logical cores.
        # Divide by 2 as a heuristic for hyperthreading.
        if os.name == "nt":
            count = max(1, count // 2)
        return count
    except Exception:
        return 1


def optimal_thread_count(n_qubits, physical_cores=None):
    """
    Compute the optimal RAYON_NUM_THREADS for a given qubit count.

    Parameters
    ----------
    n_qubits : int
        Number of data qubits in the decoding problem.
    physical_cores : int, optional
        Override for the detected physical core count.

    Returns
    -------
    int
        Recommended thread count.
    """
    if physical_cores is None:
        physical_cores = physical_core_count()

    if n_qubits < 100:
        target = 1
    elif n_qubits < 500:
        target = 2
    elif n_qubits < 2000:
        target = 4
    else:
        target = 8

    return min(target, physical_cores)


def configure_threads(n_qubits, physical_cores=None, verbose=True):
    """
    Set RAYON_NUM_THREADS before any Rayon pool initialization.

    IMPORTANT: Call this BEFORE importing or instantiating any qector decoder,
    because Rayon initializes its global thread pool on first use.

    Parameters
    ----------
    n_qubits : int
        Number of data qubits for the workload.
    physical_cores : int, optional
        Override for detected physical cores.
    verbose : bool
        Print the chosen configuration.
    """
    threads = optimal_thread_count(n_qubits, physical_cores)
    os.environ["RAYON_NUM_THREADS"] = str(threads)

    if verbose:
        cores = physical_cores or physical_core_count()
        print(
            f"[qector-thread-wrapper] n_qubits={n_qubits}  "
            f"physical_cores={cores}  "
            f"RAYON_NUM_THREADS={threads}"
        )

    return threads


def decode_with_optimal_threads(decoder_cls, n_qubits, syndromes, **decoder_kwargs):
    """
    One-shot helper: configure threads, build the decoder, decode a batch.

    Parameters
    ----------
    decoder_cls : type
        A qector decoder class (e.g. IonQSuperionDecoder, AutoDecoder).
    n_qubits : int
        Qubit count for the problem.
    syndromes : array-like
        Syndrome array or batch of syndromes.
    **decoder_kwargs
        Extra keyword arguments forwarded to the decoder constructor.

    Returns
    -------
    results : list
        Decoded correction vectors.
    """
    configure_threads(n_qubits)

    import qector_ionq
    dec = decoder_cls(**decoder_kwargs)

    if hasattr(syndromes, "ndim") and syndromes.ndim == 2:
        return [dec.decode(s) for s in syndromes]
    else:
        return dec.decode(syndromes)


if __name__ == "__main__":
    import sys

    print("QECTOR IonQ v1.7.7 Thread Allocation Table")
    print("=" * 60)
    print(f"{'Qubits':>10s}  {'Phys Cores':>10s}  {'Threads':>8s}")
    print("-" * 60)

    cores = physical_core_count()
    for q in [17, 50, 70, 100, 200, 500, 1000, 2000, 5000, 10000]:
        t = optimal_thread_count(q, cores)
        print(f"{q:>10d}  {cores:>10d}  {t:>8d}")

    print()
    print(f"Detected physical cores: {cores}")
    print(f"Logical cores (os.cpu_count): {os.cpu_count()}")
    print()

    # Live test
    configure_threads(70)
    import qector_ionq
    dec = qector_ionq.IonQSuperionDecoder()
    print(f"IonQSuperionDecoder instantiated with RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS')}")
    print("DONE")
