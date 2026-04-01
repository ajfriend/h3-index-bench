# /// script
# requires-python = ">=3.10"
# ///

"""
Benchmark H3 core API functions across two git refs.

Compiles a standalone benchmark C file against the H3 library without
modifying the repo. Builds each ref once in a temp directory, then
runs interleaved benchmark rounds.

Usage:
    uv run h3bench.py [h3_repo_path] [ref_a] [ref_b] [--rounds N]

Defaults: current directory, master, current branch, 1 round.

Multiple rounds interleave A-B-A-B to reduce system noise.
The first round is discarded as warmup when rounds > 1.

Example:
    uv run h3bench.py /path/to/h3 master vec3d-core --rounds 3
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

N_RUNS = 10

BENCHMARK_C = r"""
#include <stdio.h>
#include "benchmark.h"
#include "h3api.h"

#define N_POINTS 20
#define N_RESOLUTIONS 4
#define ITERATIONS 50000

static const LatLng points[N_POINTS] = {
    {0.659966917655, -2.1364398519396},
    {0.8527087756, -0.0405865662},
    {0.6234025842, 2.0075945568},
    {-0.5934119457, 2.5368879644},
    {0.4799655443, 0.6457718232},
    {-0.4014257280, -0.7610418886},
    {0.9679776674, -1.7453292520},
    {-1.2217304764, 0.0000000000},
    {1.2217304764, 0.0000000000},
    {0.0000000000, 0.0000000000},
    {0.0000000000, 3.1415926536},
    {0.7853981634, 1.5707963268},
    {-0.7853981634, -1.5707963268},
    {0.3490658504, -1.2217304764},
    {-0.1745329252, 0.5235987756},
    {1.0471975512, -0.5235987756},
    {-1.0471975512, 2.0943951024},
    {0.2617993878, 1.8325957146},
    {-0.8726646260, -1.0471975512},
    {0.5235987756, -2.6179938780},
};

static const int resolutions[N_RESOLUTIONS] = {0, 5, 9, 15};

int main(void) {
    H3Index cells[N_POINTS * N_RESOLUTIONS];
    int nCells = 0;
    for (int r = 0; r < N_RESOLUTIONS; r++)
        for (int p = 0; p < N_POINTS; p++)
            H3_EXPORT(latLngToCell)(&points[p], resolutions[r], &cells[nCells++]);

    H3Index edges[N_POINTS * N_RESOLUTIONS];
    int nEdges = 0;
    for (int c = 0; c < nCells; c++) {
        H3Index out[6];
        H3_EXPORT(originToDirectedEdges)(cells[c], out);
        edges[nEdges++] = out[0];
    }

    H3Index verts[N_POINTS * N_RESOLUTIONS];
    int nVerts = 0;
    for (int c = 0; c < nCells; c++) {
        H3Index out[6];
        H3_EXPORT(cellToVertexes)(cells[c], out);
        verts[nVerts++] = out[0];
    }

    H3Index h; LatLng outCoord; CellBoundary cb;

    { int total = ITERATIONS * N_POINTS * N_RESOLUTIONS;
      START_TIMER;
      for (int iter = 0; iter < ITERATIONS; iter++)
        for (int r = 0; r < N_RESOLUTIONS; r++)
          for (int p = 0; p < N_POINTS; p++)
            H3_EXPORT(latLngToCell)(&points[p], resolutions[r], &h);
      END_TIMER(d);
      printf("latLngToCell: %.4Lf us/call (%d calls)\n", d/total, total); }

    { int total = ITERATIONS * nCells;
      START_TIMER;
      for (int iter = 0; iter < ITERATIONS; iter++)
        for (int c = 0; c < nCells; c++)
          H3_EXPORT(cellToLatLng)(cells[c], &outCoord);
      END_TIMER(d);
      printf("cellToLatLng: %.4Lf us/call (%d calls)\n", d/total, total); }

    { int total = ITERATIONS * nCells;
      START_TIMER;
      for (int iter = 0; iter < ITERATIONS; iter++)
        for (int c = 0; c < nCells; c++)
          H3_EXPORT(cellToBoundary)(cells[c], &cb);
      END_TIMER(d);
      printf("cellToBoundary: %.4Lf us/call (%d calls)\n", d/total, total); }

    { int total = ITERATIONS * nEdges;
      START_TIMER;
      for (int iter = 0; iter < ITERATIONS; iter++)
        for (int e = 0; e < nEdges; e++)
          H3_EXPORT(directedEdgeToBoundary)(edges[e], &cb);
      END_TIMER(d);
      printf("directedEdgeToBoundary: %.4Lf us/call (%d calls)\n", d/total, total); }

    { int total = ITERATIONS * nVerts;
      START_TIMER;
      for (int iter = 0; iter < ITERATIONS; iter++)
        for (int v = 0; v < nVerts; v++)
          H3_EXPORT(vertexToLatLng)(verts[v], &outCoord);
      END_TIMER(d);
      printf("vertexToLatLng: %.4Lf us/call (%d calls)\n", d/total, total); }

    return 0;
}
"""


def git(repo, *args):
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=repo
    )


def get_ref_info(repo):
    branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    sha = git(repo, "rev-parse", "--short", "HEAD").stdout.strip()
    return branch, sha


def build_ref(repo, ref):
    """Checkout ref, build library in a temp dir, compile benchmark.
    Returns (bin_path, build_dir) or (None, None) on failure."""
    git(repo, "checkout", ref)
    branch, sha = get_ref_info(repo)
    print(f"  {ref}: branch={branch} commit={sha}")

    build_dir = tempfile.mkdtemp(prefix=f"h3bench-{ref}-")

    # cmake + make
    r = subprocess.run(
        f"cmake -DCMAKE_BUILD_TYPE=Release {repo} && make h3",
        shell=True, capture_output=True, text=True, cwd=build_dir,
    )
    if r.returncode != 0:
        print(f"  BUILD FAILED:\n{r.stderr[-500:]}")
        shutil.rmtree(build_dir)
        return None, None

    # write and compile benchmark
    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
        f.write(BENCHMARK_C)
        c_path = f.name

    bin_path = os.path.join(build_dir, "h3bench")
    r = subprocess.run(
        [
            "cc", "-O2",
            "-I", os.path.join(repo, "src/h3lib/include"),
            "-I", os.path.join(build_dir, "src/h3lib/include"),
            "-I", os.path.join(repo, "src/apps/applib/include"),
            c_path,
            "-L", os.path.join(build_dir, "lib"),
            "-lh3", "-lm",
            "-o", bin_path,
        ],
        capture_output=True, text=True,
    )
    os.unlink(c_path)
    if r.returncode != 0:
        print(f"  COMPILE FAILED:\n{r.stderr[-500:]}")
        shutil.rmtree(build_dir)
        return None, None

    print(f"  built: {build_dir}")
    return bin_path, build_dir


def run_bench(bin_path, n_runs):
    """Run benchmark n_runs times, return {name: min_us}."""
    best = {}
    for _ in range(n_runs):
        r = subprocess.run([bin_path], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            m = re.match(r"(\w+):\s+([\d.]+)\s+us/call", line)
            if m:
                name, us = m.group(1), float(m.group(2))
                if name not in best or us < best[name]:
                    best[name] = us
    return best


def merge_results(into, new):
    """Merge new results into accumulated results, keeping mins."""
    for name, us in new.items():
        if name not in into or us < into[name]:
            into[name] = us


def main():
    # Parse args
    args = sys.argv[1:]
    rounds = 1
    if "--rounds" in args:
        idx = args.index("--rounds")
        rounds = int(args[idx + 1])
        args = args[:idx] + args[idx + 2:]

    repo = os.path.abspath(args[0]) if len(args) > 0 else os.getcwd()
    ref_a = args[1] if len(args) > 1 else "master"
    ref_b = args[2] if len(args) > 2 else None

    if ref_b is None:
        ref_b = git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()

    # Check for clean working directory
    status = git(repo, "status", "--porcelain").stdout.strip()
    if status:
        print("Error: working directory is not clean. Commit or stash first.")
        print(status)
        return 1

    original_branch, _ = get_ref_info(repo)

    # Phase 1: Build both refs
    print("Building...")
    builds = {}
    try:
        for ref in [ref_a, ref_b]:
            bin_path, build_dir = build_ref(repo, ref)
            if bin_path is None:
                return 1
            builds[ref] = (bin_path, build_dir)
    finally:
        git(repo, "checkout", original_branch)

    print(f"\nRestored: {original_branch}")

    # Phase 2: Run interleaved rounds (no more checkouts needed)
    total_rounds = rounds + (1 if rounds > 1 else 0)
    best = {ref_a: {}, ref_b: {}}

    for round_num in range(total_rounds):
        is_warmup = rounds > 1 and round_num == 0

        print(f"\n{'='*50}")
        if is_warmup:
            print("WARMUP (discarded)")
        else:
            r_num = round_num if rounds == 1 else round_num
            print(f"Round {r_num} of {rounds}")
        print(f"{'='*50}")

        for ref in [ref_a, ref_b]:
            bin_path = builds[ref][0]
            print(f"  {ref}:")
            results = run_bench(bin_path, N_RUNS)
            for name, us in results.items():
                print(f"    {name}: {us:.4f} us/call")
            if not is_warmup:
                merge_results(best[ref], results)

    # Cleanup temp dirs
    for ref in builds:
        shutil.rmtree(builds[ref][1])

    # Report
    if best[ref_a] and best[ref_b]:
        print(f"\n{'='*50}")
        print(f"Comparison (min of {N_RUNS} runs x {rounds} rounds)")
        print(f"{'='*50}")
        print(f"{'Function':<28} {ref_a:>12} {ref_b:>12} {'Change':>10}")
        print(f"{'-'*28} {'-'*12} {'-'*12} {'-'*10}")
        for name in best[ref_a]:
            a = best[ref_a][name]
            b = best[ref_b].get(name)
            if b is not None:
                pct = (b - a) / a * 100
                sign = "+" if pct > 0 else ""
                print(f"{name:<28} {a:>10.4f}us {b:>10.4f}us {sign}{pct:>8.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
