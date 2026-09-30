"""Measure the three evaluation-budget limits for one frozen checkpoint.

Reports the scorer's wall-clock time, the process CPU time, the peak resident
set size and the uncompressed inference-asset footprint, so the numbers quoted
in REPORT.md can be reproduced on any machine.

    python measure.py --checkpoint runs/cmp-student/checkpoint.pt --split test

Run it once per checkpoint. Each invocation is a fresh process, so the peak
memory it prints belongs to that checkpoint alone.
"""
import argparse
import ctypes
import json
import sys
import time
from pathlib import Path
import torch
from common import load_data, make_model, setup
from evaluate import score


def peak_rss_gib():
    """Peak resident set size of this process, in GiB."""
    if sys.platform == 'win32':
        class Counters(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_uint32), ('PageFaultCount', ctypes.c_uint32),
                        ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        # A process handle is a pointer: without restype it is truncated to int.
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_uint32]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(),
                                         ctypes.byref(counters), counters.cb):
            raise OSError('GetProcessMemoryInfo failed')
        return counters.PeakWorkingSetSize / 1024 ** 3
    import resource
    import platform
    scale = 1 if platform.system() == 'Darwin' else 1024  # ru_maxrss is bytes on macOS, KiB elsewhere
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 1024 ** 3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--split', choices=['validation', 'test'], default='test')
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    device, precision = setup(args.device, 'fp32', args.threads)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model, _ = make_model(checkpoint['implementation'], checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'])
    data = load_data()
    process_started, wall_started = time.process_time(), time.perf_counter()
    result = score(model, *data[args.split], device, precision)
    result.pop('window_nll_nats')
    # Uncompressed footprint of every tensor the evaluator must load. Shared
    # storages (the tied embedding and output head) are counted once.
    storages = {v.untyped_storage().data_ptr(): v.untyped_storage().nbytes()
                for v in checkpoint['model'].values()}
    metrics = {'checkpoint': str(args.checkpoint), 'protocol': checkpoint['protocol'],
               'implementation': checkpoint['implementation'], 'config': checkpoint['config'],
               'split': args.split, 'device': str(device), 'precision': precision,
               'score_seconds': result['seconds'], 'wall_seconds': time.perf_counter() - wall_started,
               'process_cpu_seconds': time.process_time() - process_started,
               'peak_rss_gib': round(peak_rss_gib(), 3),
               'asset_mib_on_disk': round(args.checkpoint.stat().st_size / 1024 ** 2, 2),
               'asset_mib_uncompressed': round(sum(storages.values()) / 1024 ** 2, 2),
               'bpb': result['bpb']}
    if args.output:
        args.output.write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()