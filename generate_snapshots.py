#!/usr/bin/env python3
"""Save real simulation states and render a time-labelled particle snapshot grid."""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def read_particles(path):
    import numpy as np

    with path.open("rb") as stream:
        header = np.fromfile(stream, dtype="=i4", count=1)
        if len(header) != 1 or header[0] <= 0:
            raise ValueError(f"Invalid particle header: {path}")
        n = int(header[0])
        if path.stat().st_size != 4 + n * 28:
            raise ValueError(f"Incorrect particle file size: {path}")
        particles = np.fromfile(stream, dtype="=f4").reshape(n, 7)
    if not np.isfinite(particles).all():
        raise ValueError(f"Non-finite particle values: {path}")
    return particles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--impl", choices=["seq", "omp", "mpi", "cuda"], default="seq")
    parser.add_argument("--executable", type=Path, help="Simulator executable (default: ./nbody_<impl>)")
    parser.add_argument("--units", type=positive_int, help="Threads, MPI processes, or CUDA block size")
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--steps", nargs="+", type=int, default=[0, 40, 80, 120, 160, 200, 240, 280],
                        help="Increasing absolute step numbers; zero includes the initial state")
    parser.add_argument("--dt", type=float, default=0.01, help="Must match the simulator's compiled DT")
    parser.add_argument("--start-time", type=float, default=0.0, help="Time of the input state")
    parser.add_argument("--plane", choices=["xy", "xz", "yz"], default="xy")
    parser.add_argument("--columns", type=positive_int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("results/snapshots"))
    parser.add_argument("--figure", type=Path, default=Path("figures/snapshots"), help="Output stem for PNG and PDF")
    args = parser.parse_args()
    if not args.steps or args.steps[0] < 0 or any(b <= a for a, b in zip(args.steps, args.steps[1:])):
        parser.error("--steps must be nonnegative and strictly increasing")
    if not math.isfinite(args.dt) or args.dt <= 0 or not math.isfinite(args.start_time):
        parser.error("--dt must be finite and positive; --start-time must be finite")

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    initial = read_particles(args.input)
    executable = args.executable or Path(f"nbody_{args.impl}" + (".exe" if sys.platform == "win32" else ""))
    executable = executable.resolve()
    if args.steps[-1] > 0 and not executable.is_file():
        parser.error(f"Simulator not found: {executable}; build it or pass --executable")
    units = args.units or (256 if args.impl == "cuda" else 4)
    targets = [args.output_dir / f"step_{step:08d}.bin" for step in args.steps]
    metadata_path = args.output_dir / "metadata.json"
    figures = [Path(str(args.figure) + suffix) for suffix in (".png", ".pdf")]
    for path in [*targets, metadata_path, *figures]:
        if path.exists():
            parser.error(f"Output already exists: {path}; choose a new output directory/figure stem")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    previous_path, previous_step = args.input.resolve(), 0
    frames, records = [], []
    axes = ["xyz".index(axis) + 1 for axis in args.plane]
    for step, target in zip(args.steps, targets):
        if step == 0:
            shutil.copyfile(previous_path, target)
        else:
            command = [str(executable), str(previous_path), str(step - previous_step)]
            if args.impl in ("omp", "cuda"):
                command.append(str(units))
            command.append(str(target.resolve()))
            if args.impl == "mpi":
                command = [args.mpirun, "-np", str(units), *command]
            print(f"Generating step {step}...", flush=True)
            subprocess.run(command, check=True)
        particles = read_particles(target)
        if particles.shape != initial.shape or not np.array_equal(particles[:, 0], initial[:, 0]):
            raise ValueError(f"Particle count or masses changed: {target}")
        frames.append(particles[:, axes])
        records.append({"step": step, "time": args.start_time + step * args.dt, "file": target.name})
        previous_path, previous_step = target.resolve(), step
    metadata = {"input": str(args.input.resolve()), "implementation": args.impl,
                "executable": str(executable), "units": units if args.impl != "seq" else 1,
                "dt": args.dt, "start_time": args.start_time, "particles": len(initial),
                "plane": args.plane, "snapshots": records}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    # One fixed, square view containing every particle at every sampled time.
    lower = np.min([frame.min(axis=0) for frame in frames], axis=0)
    upper = np.max([frame.max(axis=0) for frame in frames], axis=0)
    center = (lower + upper) / 2
    radius = max(float(np.max(upper - lower)) * 0.55, 1e-6)
    columns = min(args.columns, len(frames))
    rows = math.ceil(len(frames) / columns)
    fig, panels = plt.subplots(rows, columns, figsize=(3 * columns, 3.3 * rows), squeeze=False)
    for panel, frame, record in zip(panels.flat, frames, records):
        panel.scatter(frame[:, 0], frame[:, 1], s=0.6, c="black", linewidths=0, rasterized=True)
        panel.set(xlim=(center[0] - radius, center[0] + radius),
                  ylim=(center[1] - radius, center[1] + radius), aspect="equal", xticks=[], yticks=[])
        panel.set_title(f"Time = {record['time']:.6g}", fontsize=11, pad=10)
    for panel in list(panels.flat)[len(frames):]:
        panel.set_visible(False)
    fig.text(0.5, 0.015, f"N = {len(initial):,}  |  {args.impl.upper()}  |  {args.plane.upper()} projection  |  N-body time units",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=2)
    for path in figures:
        fig.savefig(path, dpi=250, facecolor="white")
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
