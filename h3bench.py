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
import subprocess
import tempfile

import click

BENCHMARK_C_TEMPLATE = r"""
#include <stdio.h>
#include "benchmark.h"
#include "h3api.h"

#define N_POINTS 20
#define N_RESOLUTIONS 4
#define ITERATIONS {iterations}

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

    {{ int total = ITERATIONS * N_POINTS * N_RESOLUTIONS;
      START_TIMER;
      for (int iter = 0; iter < ITERATIONS; iter++)
        for (int r = 0; r < N_RESOLUTIONS; r++)
          for (int p = 0; p < N_POINTS; p++)
            H3_EXPORT(latLngToCell)(&points[p], resolutions[r], &h);
      END_TIMER(d);
      printf("latLngToCell: %.4Lf us/call (%d calls)\\n", d/total, total); }}

    {{ int total = ITERATIONS * nCells;
      START_TIMER;
      for (int iter = 0; iter < ITERATIONS; iter++)
        for (int c = 0; c < nCells; c++)
          H3_EXPORT(cellToLatLng)(cells[c], &outCoord);
      END_TIMER(d);
      printf("cellToLatLng: %.4Lf us/call (%d calls)\\n", d/total, total); }}

    {{ int total = ITERATIONS * nCells;
      START_TIMER;
      for (int iter = 0; iter < ITERATIONS; iter++)
        for (int c = 0; c < nCells; c++)
          H3_EXPORT(cellToBoundary)(cells[c], &cb);
      END_TIMER(d);
      printf("cellToBoundary: %.4Lf us/call (%d calls)\\n", d/total, total); }}

    {{ int total = ITERATIONS * nEdges;
      START_TIMER;
      for (int iter = 0; iter < ITERATIONS; iter++)
        for (int e = 0; e < nEdges; e++)
          H3_EXPORT(directedEdgeToBoundary)(edges[e], &cb);
      END_TIMER(d);
      printf("directedEdgeToBoundary: %.4Lf us/call (%d calls)\\n", d/total, total); }}

    {{ int total = ITERATIONS * nVerts;
      START_TIMER;
      for (int iter = 0; iter < ITERATIONS; iter++)
        for (int v = 0; v < nVerts; v++)
          H3_EXPORT(vertexToLatLng)(verts[v], &outCoord);
      END_TIMER(d);
      printf("vertexToLatLng: %.4Lf us/call (%d calls)\\n", d/total, total); }}

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
    Returns (bin_path, worktree_dir)."""
    sha = get_sha(repo, ref)
    click.echo(f"  {ref} ({sha})")

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
    c_source = BENCHMARK_C_TEMPLATE.format(iterations=iterations)
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

    return bin_path, worktree_dir


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
    for name, us in new.items():
        if name not in into or us < into[name]:
            into[name] = us


@click.command()
@click.argument("repo", default=".", type=click.Path(exists=True))
@click.argument("ref_a", default="master")
@click.argument("ref_b", default=None, required=False)
@click.option("--rounds", default=1, help="Number of interleaved A-B rounds (min over rounds).", show_default=True)
@click.option("--runs", default=10, help="Runs per round (min over runs).", show_default=True)
@click.option("--iterations", default=50000, help="Inner loop iterations in C.", show_default=True)
def bench(repo, ref_a, ref_b, rounds, runs, iterations):
    """Benchmark H3 core API across two git refs.

    Uses git worktrees so the main checkout is never modified.
    Builds each ref once, then runs interleaved benchmark rounds.
    The first round is discarded as warmup when rounds > 1.
    """
    repo = os.path.abspath(repo)

    if ref_b is None:
        r = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
        ref_b = r.stdout.strip()

    # Build both refs in worktrees
    click.echo("Building...")
    builds = {}
    try:
        for ref in [ref_a, ref_b]:
            bin_path, worktree_dir = build_ref(repo, ref, iterations)
            builds[ref] = (bin_path, worktree_dir)

        # Run interleaved rounds
        total_rounds = rounds + (1 if rounds > 1 else 0)
        best = {ref_a: {}, ref_b: {}}

        for round_num in range(total_rounds):
            is_warmup = rounds > 1 and round_num == 0

            click.echo(f"\n{'='*50}")
            if is_warmup:
                click.echo("WARMUP (discarded)")
            else:
                click.echo(f"Round {round_num} of {rounds}")
            click.echo(f"{'='*50}")

            for ref in [ref_a, ref_b]:
                bin_path = builds[ref][0]
                click.echo(f"  {ref}:")
                results = run_bench(bin_path, runs)
                for name, us in results.items():
                    click.echo(f"    {name}: {us:.4f} us/call")
                if not is_warmup:
                    merge_results(best[ref], results)

        # Report
        if best[ref_a] and best[ref_b]:
            click.echo(f"\n{'='*50}")
            click.echo(f"Comparison (min of {runs} runs x {rounds} rounds)")
            click.echo(f"{'='*50}")
            click.echo(f"{'Function':<28} {ref_a:>12} {ref_b:>12} {'Change':>10}")
            click.echo(f"{'-'*28} {'-'*12} {'-'*12} {'-'*10}")
            for name in best[ref_a]:
                a = best[ref_a][name]
                b = best[ref_b].get(name)
                if b is not None:
                    pct = (b - a) / a * 100
                    sign = "+" if pct > 0 else ""
                    click.echo(
                        f"{name:<28} {a:>10.4f}us {b:>10.4f}us {sign}{pct:>8.1f}%"
                    )

    finally:
        # Cleanup worktrees
        for ref in builds:
            worktree_dir = builds[ref][1]
            git(repo, "worktree", "remove", "--force", worktree_dir)


if __name__ == "__main__":
    bench()
