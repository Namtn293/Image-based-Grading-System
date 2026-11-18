"""Nghiệm thu Bảng 6.1: mọi phép đối chứng Scratch vs. OpenCV của TV2 phải đạt PASS."""

from tv2.benchmark import run_benchmark


def test_all_benchmarks_pass():
    rows = run_benchmark(save_outputs=False)
    assert len(rows) == 4
    failed = [r["algorithm"] for r in rows if r["status"] != "PASS"]
    assert not failed, f"Benchmark FAIL: {failed}"
