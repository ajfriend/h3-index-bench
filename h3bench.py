# /// script
# requires-python = ">=3.10"
# dependencies = ["click"]
# ///

"""
Benchmark H3 core API functions across two git refs.

Compiles a standalone benchmark C file against the H3 library without
modifying the repo. Uses git worktrees so the main checkout is never
touched.
"""

import os
import re
import shutil
import statistics
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import click

BENCHMARK_C_TEMPLATE = r"""
#include <stdio.h>
#include "benchmark.h"
#include "h3api.h"

#define N_POINTS 20
#define N_RESOLUTIONS 4
#define ITERATIONS {iterations}
#define ITERATIONS_FORWARD {iterations_forward}

static const LatLng points[N_POINTS] = {{
    {{0.659966917655, -2.1364398519396}},
    {{0.8527087756, -0.0405865662}},
    {{0.6234025842, 2.0075945568}},
    {{-0.5934119457, 2.5368879644}},
    {{0.4799655443, 0.6457718232}},
    {{-0.4014257280, -0.7610418886}},
    {{0.9679776674, -1.7453292520}},
    {{-1.2217304764, 0.0000000000}},
    {{1.2217304764, 0.0000000000}},
    {{0.0000000000, 0.0000000000}},
    {{0.0000000000, 3.1415926536}},
    {{0.7853981634, 1.5707963268}},
    {{-0.7853981634, -1.5707963268}},
    {{0.3490658504, -1.2217304764}},
    {{-0.1745329252, 0.5235987756}},
    {{1.0471975512, -0.5235987756}},
    {{-1.0471975512, 2.0943951024}},
    {{0.2617993878, 1.8325957146}},
    {{-0.8726646260, -1.0471975512}},
    {{0.5235987756, -2.6179938780}},
}};

static const int resolutions[N_RESOLUTIONS] = {{0, 5, 9, 15}};

int main(void) {{
    H3Index cells[N_POINTS * N_RESOLUTIONS];
    int nCells = 0;
    for (int r = 0; r < N_RESOLUTIONS; r++)
        for (int p = 0; p < N_POINTS; p++)
            H3_EXPORT(latLngToCell)(&points[p], resolutions[r], &cells[nCells++]);

    H3Index edges[N_POINTS * N_RESOLUTIONS];
    int nEdges = 0;
    for (int c = 0; c < nCells; c++) {{
        H3Index out[6];
        H3_EXPORT(originToDirectedEdges)(cells[c], out);
        edges[nEdges++] = out[0];
    }}

    H3Index verts[N_POINTS * N_RESOLUTIONS];
    int nVerts = 0;
    for (int c = 0; c < nCells; c++) {{
        H3Index out[6];
        H3_EXPORT(cellToVertexes)(cells[c], out);
        verts[nVerts++] = out[0];
    }}

    H3Index h; LatLng outCoord; CellBoundary cb;

    {{ int total = ITERATIONS_FORWARD * N_POINTS * N_RESOLUTIONS;
      START_TIMER;
      for (int iter = 0; iter < ITERATIONS_FORWARD; iter++)
        for (int r = 0; r < N_RESOLUTIONS; r++)
          for (int p = 0; p < N_POINTS; p++)
            H3_EXPORT(latLngToCell)(&points[p], resolutions[r], &h);
      END_TIMER(d);
      printf("latLngToCell: %.4Lf us/call (%d calls)\n", d/total, total); }}

    {{ int total = ITERATIONS * nCells;
      START_TIMER;
      for (int iter = 0; iter < ITERATIONS; iter++)
        for (int c = 0; c < nCells; c++)
          H3_EXPORT(cellToLatLng)(cells[c], &outCoord);
      END_TIMER(d);
      printf("cellToLatLng: %.4Lf us/call (%d calls)\n", d/total, total); }}

    {{ int total = ITERATIONS * nCells;
      START_TIMER;
      for (int iter = 0; iter < ITERATIONS; iter++)
        for (int c = 0; c < nCells; c++)
          H3_EXPORT(cellToBoundary)(cells[c], &cb);
      END_TIMER(d);
      printf("cellToBoundary: %.4Lf us/call (%d calls)\n", d/total, total); }}

    {{ int total = ITERATIONS * nEdges;
      START_TIMER;
      for (int iter = 0; iter < ITERATIONS; iter++)
        for (int e = 0; e < nEdges; e++)
          H3_EXPORT(directedEdgeToBoundary)(edges[e], &cb);
      END_TIMER(d);
      printf("directedEdgeToBoundary: %.4Lf us/call (%d calls)\n", d/total, total); }}

    {{ int total = ITERATIONS * nVerts;
      START_TIMER;
      for (int iter = 0; iter < ITERATIONS; iter++)
        for (int v = 0; v < nVerts; v++)
          H3_EXPORT(vertexToLatLng)(verts[v], &outCoord);
      END_TIMER(d);
      printf("vertexToLatLng: %.4Lf us/call (%d calls)\n", d/total, total); }}

    return 0;
}}
"""


def git(repo, *args):
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=repo
    )


def get_sha(repo, ref):
    r = git(repo, "rev-parse", "--short", ref)
    if r.returncode != 0:
        raise click.ClickException(f"unknown ref: {ref}")
    return r.stdout.strip()


def build_ref(repo, ref, iterations):
    """Create worktree for ref, build library, compile benchmark.
    Returns (sha, bin_path, worktree_dir)."""
    sha = get_sha(repo, ref)

    worktree_dir = tempfile.mkdtemp(prefix=f"h3bench-{ref}-")
    r = git(repo, "worktree", "add", "--detach", worktree_dir, ref)
    if r.returncode != 0:
        shutil.rmtree(worktree_dir)
        raise click.ClickException(f"worktree failed for {ref}: {r.stderr.strip()}")

    # Build library
    build_dir = os.path.join(worktree_dir, "build")
    os.makedirs(build_dir)
    r = subprocess.run(
        f"cmake -DCMAKE_BUILD_TYPE=Release {worktree_dir} && make h3",
        shell=True, capture_output=True, text=True, cwd=build_dir,
    )
    if r.returncode != 0:
        git(repo, "worktree", "remove", "--force", worktree_dir)
        raise click.ClickException(f"build failed for {ref}:\n{r.stderr[-500:]}")

    # Compile benchmark
    c_source = BENCHMARK_C_TEMPLATE.format(
        iterations=iterations, iterations_forward=iterations * 5
    )
    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
        f.write(c_source)
        c_path = f.name

    bin_path = os.path.join(build_dir, "h3bench")
    r = subprocess.run(
        [
            "cc", "-O2",
            "-I", os.path.join(worktree_dir, "src/h3lib/include"),
            "-I", os.path.join(build_dir, "src/h3lib/include"),
            "-I", os.path.join(worktree_dir, "src/apps/applib/include"),
            c_path,
            "-L", os.path.join(build_dir, "lib"),
            "-lh3", "-lm",
            "-o", bin_path,
        ],
        capture_output=True, text=True,
    )
    os.unlink(c_path)
    if r.returncode != 0:
        git(repo, "worktree", "remove", "--force", worktree_dir)
        raise click.ClickException(f"compile failed for {ref}:\n{r.stderr[-500:]}")

    return sha, bin_path, worktree_dir


def run_bench(bin_path):
    """Run benchmark once, return {name: us}."""
    results = {}
    r = subprocess.run([bin_path], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        m = re.match(r"(\w+):\s+([\d.]+)\s+us/call", line)
        if m:
            results[m.group(1)] = float(m.group(2))
    return results


def collect_results(into, new):
    """Append new sample into accumulated sample lists."""
    for name, us in new.items():
        into.setdefault(name, []).append(us)


@click.command()
@click.argument("repo", default=".", type=click.Path(exists=True))
@click.argument("ref_a", default="master")
@click.argument("ref_b", default=None, required=False)
@click.option("--samples", default=20, help="Number of interleaved A-B sample pairs.", show_default=True)
@click.option("--iterations", default=10000, help="Inner loop iterations in C.", show_default=True)
def bench(repo, ref_a, ref_b, samples, iterations):
    """Benchmark H3 core API across two git refs.

    Uses git worktrees so the main checkout is never modified.
    Builds each ref once, then runs interleaved A-B samples.
    The first sample pair is discarded as warmup when samples > 1.
    Reports the median.
    """
    t0 = time.monotonic()
    repo = os.path.abspath(repo)

    if ref_b is None:
        r = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
        ref_b = r.stdout.strip()

    # Build both refs in worktrees (in parallel)
    click.echo("Building...")
    builds = {}
    try:
        shas = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                ref: pool.submit(build_ref, repo, ref, iterations)
                for ref in [ref_a, ref_b]
            }
            for ref in [ref_a, ref_b]:
                sha, bin_path, worktree_dir = futures[ref].result()
                builds[ref] = (bin_path, worktree_dir)
                shas[ref] = sha
                click.echo(f"  {ref} ({sha})")

        # Run interleaved A-B samples
        total_pairs = samples + (1 if samples > 1 else 0)
        all_samples = {ref_a: {}, ref_b: {}}

        for pair_num in range(total_pairs):
            is_warmup = samples > 1 and pair_num == 0

            click.secho(f"\n{'='*50}", dim=True)
            if is_warmup:
                click.secho("WARMUP (discarded)", dim=True)
            else:
                click.echo(f"Sample {pair_num} of {samples}")
            click.secho(f"{'='*50}", dim=True)

            for ref in [ref_a, ref_b]:
                bin_path = builds[ref][0]
                click.echo(f"  {ref}:")
                result = run_bench(bin_path)
                for name, us in result.items():
                    click.echo(f"    {name}: {us:.4f} us/call")
                if not is_warmup:
                    collect_results(all_samples[ref], result)

        # Report using median
        if all_samples[ref_a] and all_samples[ref_b]:
            click.echo()
            click.secho(f"{'='*60}", bold=True)
            click.secho(f"Comparison (median of {samples} samples)", bold=True)
            click.echo(f"  {ref_a} = {shas[ref_a]}")
            click.echo(f"  {ref_b} = {shas[ref_b]}")
            click.secho(f"{'='*60}", bold=True)
            click.echo(f"{'Function':<28} {ref_a:>12} {ref_b:>12} {'Change':>10}")
            click.secho(f"{'-'*28} {'-'*12} {'-'*12} {'-'*10}", dim=True)
            for name in all_samples[ref_a]:
                a = statistics.median(all_samples[ref_a][name])
                b_samples = all_samples[ref_b].get(name)
                if b_samples is not None:
                    b = statistics.median(b_samples)
                    pct = (b - a) / a * 100
                    if pct < -1:
                        color = "green"
                    elif pct > 1:
                        color = "red"
                    else:
                        color = None
                    sign = "+" if pct > 0 else ""
                    pct_str = click.style(f"{sign}{pct:>8.1f}%", fg=color)
                    click.echo(
                        f"{name:<28} {a:>10.4f}us {b:>10.4f}us {pct_str}"
                    )

    finally:
        # Cleanup worktrees
        for ref in builds:
            worktree_dir = builds[ref][1]
            git(repo, "worktree", "remove", "--force", worktree_dir)

    elapsed = time.monotonic() - t0
    click.secho(f"\nCompleted in {elapsed:.0f}s", dim=True)


if __name__ == "__main__":
    bench()
