# h3bench

Benchmark [H3](https://github.com/uber/h3) core API functions across two git refs.

Run directly from GitHub:

```
uv run https://raw.githubusercontent.com/ajfriend/h3-index-bench/main/bench.py ~/work/h3 master vec3d-core
```

Builds each ref in an isolated git worktree, then runs interleaved A/B samples
of a C microbenchmark and reports the median time per call with a live TUI.

## Benchmarked functions

- `latLngToCell`
- `cellToLatLng`
- `cellToBoundary`
- `directedEdgeToBoundary`
- `vertexToLatLng`

## Usage

```
uv run h3bench.py <h3-repo> <ref-a> [ref-b]
```

- `<h3-repo>`: path to a local H3 git checkout (default: `.`)
- `<ref-a>`: first git ref to compare (default: `master`)
- `[ref-b]`: second git ref (default: current `HEAD`)

### Options

| Option         | Default | Description                            |
|----------------|---------|----------------------------------------|
| `--samples`    | 20      | Number of interleaved A/B sample pairs |
| `--iterations` | 10000   | Inner loop iterations in C benchmark   |

### Example from a local clone

```
uv run bench.py ~/src/h3 v4.1.0 main --samples 30
```
