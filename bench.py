# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "rich", "tabulate"]
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
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import click
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text
from tabulate import tabulate

console = Console()

SPARK_CHARS = "▁▂▃▄▅▆▇█"

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


SCRIPT_REPO = "ajfriend/h3-index-bench"


def git(repo, *args):
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=repo
    )


def script_sha():
    """Get the commit SHA of this script, via local git or the GitHub API."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    r = git(script_dir, "rev-parse", "--short", "HEAD")
    if r.returncode == 0:
        return r.stdout.strip()
    try:
        url = f"https://api.github.com/repos/{SCRIPT_REPO}/commits/main"
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.sha"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.read(7).decode()
    except Exception:
        return "unknown"


def get_sha(repo, ref):
    r = git(repo, "rev-parse", "--short", ref)
    if r.returncode != 0:
        raise click.ClickException(f"unknown ref: {ref}")
    return r.stdout.strip()


def build_ref(repo, ref, iterations):
    """Create worktree for ref, build library, compile benchmark."""
    sha = get_sha(repo, ref)

    worktree_dir = tempfile.mkdtemp(prefix=f"h3bench-{ref.replace('/', '-')}-")
    r = git(repo, "worktree", "add", "--detach", worktree_dir, ref)
    if r.returncode != 0:
        shutil.rmtree(worktree_dir)
        raise click.ClickException(f"worktree failed for {ref}: {r.stderr.strip()}")

    build_dir = os.path.join(worktree_dir, "build")
    os.makedirs(build_dir)
    r = subprocess.run(
        f"cmake -DCMAKE_BUILD_TYPE=Release {worktree_dir} && make h3",
        shell=True, capture_output=True, text=True, cwd=build_dir,
    )
    if r.returncode != 0:
        git(repo, "worktree", "remove", "--force", worktree_dir)
        raise click.ClickException(f"build failed for {ref}:\n{r.stderr[-500:]}")

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


def sparkline(values, max_width=30):
    """Generate a sparkline string, downsampling if needed."""
    if not values:
        return ""
    if len(values) > max_width:
        values = values[-max_width:]
    mn, mx = min(values), max(values)
    rng = mx - mn if mx > mn else 1
    return "".join(
        SPARK_CHARS[min(int((v - mn) / rng * (len(SPARK_CHARS) - 1)), len(SPARK_CHARS) - 1)]
        for v in values
    )


def make_live_table(ref_a, ref_b, shas, all_samples, func_names, completed, total):
    """Build the rich table for live display."""
    table = Table(title=f"h3bench — sample {completed}/{total}", border_style="dim")
    table.add_column("Function", style="bold", min_width=26)
    table.add_column(f"{ref_a}\n({shas[ref_a]})", justify="right", min_width=10)
    spark_width = min(total, 30)
    table.add_column("", justify="left", min_width=spark_width, max_width=spark_width)
    table.add_column(f"{ref_b}\n({shas[ref_b]})", justify="right", min_width=10)
    table.add_column("", justify="left", min_width=spark_width, max_width=spark_width)
    table.add_column("Change", justify="right", min_width=9)

    for name in func_names:
        a_samples = all_samples.get(ref_a, {}).get(name, [])
        b_samples = all_samples.get(ref_b, {}).get(name, [])

        a_med = f"{statistics.median(a_samples):.4f}" if a_samples else "—"
        b_med = f"{statistics.median(b_samples):.4f}" if b_samples else "—"

        a_spark = Text(sparkline(a_samples), style="cyan")
        b_spark = Text(sparkline(b_samples), style="magenta")

        if a_samples and b_samples:
            a_val = statistics.median(a_samples)
            b_val = statistics.median(b_samples)
            pct = (b_val - a_val) / a_val * 100
            sign = "+" if pct > 0 else ""
            if pct < -1:
                style = "bold green"
            elif pct > 1:
                style = "bold red"
            else:
                style = "dim"
            change = Text(f"{sign}{pct:.1f}%", style=style)
        else:
            change = Text("—", style="dim")

        table.add_row(name, a_med, a_spark, b_med, b_spark, change)

    return table


@click.command()
@click.argument("repo", default=".", type=click.Path(exists=True))
@click.argument("ref_a", default="master")
@click.argument("ref_b", default=None, required=False)
@click.option("--samples", default=20, help="Number of interleaved A-B sample pairs.", show_default=True)
@click.option("--iterations", default=10000, help="Inner loop iterations in C.", show_default=True)
@click.option("--markdown", is_flag=True, help="Print final table as GitHub-flavored markdown.")
def bench(repo, ref_a, ref_b, samples, iterations, markdown):
    """Benchmark H3 core API across two git refs.

    Uses git worktrees so the main checkout is never modified.
    Builds each ref once, then runs interleaved A-B samples.
    Reports the median.
    """
    t0 = time.monotonic()
    repo = os.path.abspath(repo)

    if ref_b is None:
        r = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
        ref_b = r.stdout.strip()

    # Warn if the repo has uncommitted changes (worktrees use committed state)
    r = git(repo, "status", "--porcelain")
    if r.stdout.strip():
        console.print(
            "[bold yellow]Warning:[/bold yellow] repo has uncommitted changes "
            "— worktrees will use the last committed state."
        )

    # Build both refs in worktrees (in parallel)
    console.print("[bold]Building...[/bold]")
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
                console.print(f"  {ref} ({sha})")

        # Run interleaved A-B samples with live TUI
        all_samples = {ref_a: {}, ref_b: {}}
        func_names = None
        completed = 0

        with Live(console=console, refresh_per_second=2) as live:
            for pair_num in range(samples):
                for ref in [ref_a, ref_b]:
                    bin_path = builds[ref][0]
                    result = run_bench(bin_path)

                    if func_names is None:
                        func_names = list(result.keys())

                    for name, us in result.items():
                        all_samples[ref].setdefault(name, []).append(us)

                completed += 1
                live.update(
                    make_live_table(
                        ref_a, ref_b, shas, all_samples,
                        func_names or [], completed, samples
                        )
                    )

        # Final report
        if all_samples[ref_a] and all_samples[ref_b]:
            console.print()
            table = Table(
                title=f"[bold]Comparison (median of {samples} samples)[/bold]",
                caption=f"{ref_a}={shas[ref_a]}  {ref_b}={shas[ref_b]}",
                border_style="bold",
            )
            table.add_column("Function", style="bold", min_width=26)
            table.add_column(ref_a, justify="right")
            table.add_column(ref_b, justify="right")
            table.add_column("Change", justify="right")

            md_rows = []
            for name in func_names:
                a = statistics.median(all_samples[ref_a][name])
                b_vals = all_samples[ref_b].get(name)
                if b_vals is not None:
                    b = statistics.median(b_vals)
                    pct = (b - a) / a * 100
                    sign = "+" if pct > 0 else ""
                    if pct < -1:
                        style = "bold green"
                    elif pct > 1:
                        style = "bold red"
                    else:
                        style = ""
                    change = f"[{style}]{sign}{pct:.1f}%[/{style}]" if style else f"{sign}{pct:.1f}%"
                    table.add_row(name, f"{a:.4f}us", f"{b:.4f}us", change)
                    md_rows.append([name, f"{a:.4f}us", f"{b:.4f}us", f"{sign}{pct:.1f}%"])

            console.print(table)

            if markdown:
                bench_sha = script_sha()
                console.print()
                print(tabulate(
                    md_rows,
                    headers=["Function", ref_a, ref_b, "Change"],
                    tablefmt="github",
                    colalign=("left", "right", "right", "right"),
                ))
                print(f"\n{ref_a}={shas[ref_a]}  {ref_b}={shas[ref_b]}  bench={bench_sha}")

    finally:
        for ref in builds:
            worktree_dir = builds[ref][1]
            git(repo, "worktree", "remove", "--force", worktree_dir)

    elapsed = time.monotonic() - t0
    console.print(f"\n[dim]Completed in {elapsed:.0f}s[/dim]")


if __name__ == "__main__":
    bench()
