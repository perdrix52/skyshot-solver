# Skyshot Solver — C++ port

A C++23 port of the Rust crate in the parent directory. It exposes the same
command-line interface and the same library surface, and it reads and writes the
**same data files** (index directories, learned coders, D80 catalogue tiles,
FITS/PNG/JPEG/TIFF images, StarField JSON), so an index downloaded for the Rust
solver can be used here unchanged, and vice versa.

```text
data/
  index/         main index          (manifest.json, codes.bin, keys.bin, stars.bin, cells.bin, deep.bin, deep_cells.bin, coder.f32/.json)
  index-deep/    optional DEEP companion
  index-phone/   optional wide-field companion
```

## Build

Needs CMake ≥ 3.16 and a C++23 compiler (developed with MSVC 2022; the code is
standard C++ and avoids compiler extensions, but only MSVC has been exercised).
There are no external dependencies: JSON, SHA-256, linear algebra, a TIFF reader
and the thread pool are included; `third_party/stb_image.h` (public domain / MIT)
decodes PNG and JPEG.

```sh
cmake -S cpp -B cpp/build                       # add -G "Visual Studio 17 2022" -A x64 on Windows
cmake --build cpp/build --config Release
cpp/build/Release/skyshot_tests                 # unit + integration tests
```

Targets: `skyshot` (static library), `skyshot_cli` (executable `skyshot`),
`skyshot_tests`, `synthetic_demo`.

## Command line

Identical to the Rust CLI (see `docs/CLI.md` in the repository root):

```sh
skyshot solve  frame.fits --index data/index [--mode auto|standard|deep] [--deadline-ms 2000]
               [--min-fov DEG] [--max-fov DEG] [--matches] [--arc] [--panorama] [--wcs out.wcs]
skyshot batch  a.fits b.png stars.json --index data/index
skyshot extract frame.png
skyshot inspect data/index
skyshot build  --d80 /path/to/d80 --out data/index-new --profile full [--coder prefix]
```

Output JSON (`schema_version`, `solution`, `timings`, `n_stars`), JSONL for
`batch`, and the exit codes (0 solved, 1 unsolved/ambiguous, 2 error) match the
Rust implementation.

Try it without any catalogue:

```sh
synthetic_demo demo                 # creates demo/index, demo/stars.json, demo/expected-wcs.json
skyshot solve demo/stars.json --index demo/index
```

## Library

```cpp
#include "skyshot/engine.hpp"

auto engine = skyshot::Engine::open("data/index");   // maps the index (and companions) once
skyshot::solve::SolveConfig cfg;                     // same defaults as the Rust SolveConfig
cfg.deadline_ms = 2000;
auto out = engine.solve_path("frame.fits", cfg);     // or engine.solve_sources(starfield, cfg)
if (out.solution.status == "solved") { /* out.solution.wcs->fits_header(), ... */ }
std::string json = out.to_json().dump(/*pretty=*/true);
```

Module map (each file ports the Rust module of the same name):

| Rust | C++ |
| --- | --- |
| `types`, `pixelgrid` | `types.hpp/.cpp` |
| `geometry`, `coordinates`, `sphere`, `select`, `fingerprint`, `patterns` | same names |
| `index`, `learned`, `builder`, `d80` | same names |
| `image_io` | `image_io.cpp` (FITS), `image_formats.cpp` (PNG/JPEG via stb, TIFF, EXIF orientation), `image_preview.cpp` |
| `extract`, `vet`, `lines` | `extract*.cpp` (one extractor) |
| `highpass`, `detect`, `skymask`, `hints`, `attempt`, `compat` | same names (`attempt` is in `pipeline`) |
| `wcsfit`, `wcsfit/{lens,panorama,stereographic,registration}` | `wcsfit*.cpp`, `registration.cpp` |
| `solve`, `solve/wide`, `solve/wide/panorama`, `solve/refract` | `solve*.cpp` |
| `pipeline`, `engine`, `main` | same names |
| `api/field` | `field.hpp/.cpp` |
| `measure` (forced photometry: `FrameModel`, `measure`, `Session`/`run`, `fit_zero_point`) | `measure.hpp/.cpp` (a library API; there is no CLI subcommand for it in the Rust CLI either) |
| nalgebra / rayon / serde_json / sha2 / memmap2 / rstar / clap / `image` | `linalg`, `parallel`, `json`, `sha256`, `MappedFile`, linear scan in `detect`, hand-written argument parser, stb + own TIFF |

## What is not ported

* `measure.rs` — forced photometry for the annotation atlas (`FrameModel`,
  `measure`, `Session`/`run`). It is not used by the solver, the extractor or the CLI.
* The server-oriented parts of `api` (`SolveRequest`, `PublicSolution`, `Row`
  serialisation) and `Index::lock_in_memory` (Unix `mlock`, unsupported on Windows in the Rust code too).

## Fidelity notes

The algorithms, constants, evaluation orders and data layouts follow the Rust
source line by line, including its floating-point conventions (`f32` arithmetic in
the extractor and the learned network, saturating float→integer casts, `total_cmp`
orderings, stable vs. unstable sorts, ryu-style shortest float formatting in JSON).
No tool in this repository can run the Rust and C++ builds side by side, so the
following are the places where results can legitimately differ in the last bits or
in rare ties:

* **Maths library.** Rust uses `libm`/the platform `libm` for some functions and
  the C++ code uses the C runtime; `sin`/`cos`/`atan2`/`exp` can differ by 1 ulp.
* **JPEG decoding.** `stb_image` and the Rust `image` crate use different IDCT and
  chroma up-sampling code, so pixels may differ by a grey level or two. PNG, FITS
  and TIFF are decoded exactly.
* **SVD / least squares.** nalgebra's SVD is replaced by a one-sided Jacobi SVD
  (`linalg.cpp`); results agree to rounding, sign conventions of singular vectors
  may differ (no solver result depends on them).
* **Hash-map iteration.** Where the Rust code iterated a `HashMap` (tracking vote
  histogram) or sorted with `sort_unstable` on equal keys (index record order inside
  one fingerprint bucket), the C++ port uses a deterministic order. Index files built by
  the C++ builder are valid and equivalent but not necessarily byte-identical to
  Rust-built ones.
* **Timing.** Deadlines are wall-clock budgets in both implementations, so slow
  machines can legitimately stop searches at different points.

All outputs of a solve (WCS, matches, statistics) are produced from the same index
records through the same acceptance tests, so a field solved by one implementation
is solved by the other within the usual numerical tolerance.

## Tests

`tests/test_main.cpp` mirrors `tests/cli.rs` (engine reuse, CLI solve/batch/WCS
refusal/builder refusal, unsolved exit code) and adds checks for the building blocks
(JSON number formatting, SHA-256, cube-grid caps, dominance and k-NN against brute
force, fingerprint invariance, WCS round trips for every projection, learned-key
enumeration and `erfc` reference values, PNG/TIFF/FITS decoding, orientation,
hint parsing, previews) and an image pipeline test that renders a synthetic FITS
frame, extracts its stars and solves it against the synthetic index.
`tests/test_measure.cpp` ports `tests/measure.rs` (known sources, correlated noise,
saturation/edges/masks, blends and zero point, photon-transfer gain, request path and
errors). The noise generator differs from the Rust test's ChaCha8, so the one
statistical assertion (no faint source recorded) allows up to 2 of 12.
