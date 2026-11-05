#!/usr/bin/env python3
"""Draw architecture diagrams matching this project's actual simulation code."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('figures/diagrams'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})

    def canvas(title, subtitle):
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.set(xlim=(0, 12), ylim=(0, 7))
        ax.axis('off')
        ax.text(.3, 6.6, title, fontsize=20, weight='bold', color='#172554')
        ax.text(.3, 6.12, subtitle, fontsize=11, color='#475569')
        return fig, ax

    def box(ax, x, y, w, h, text, color='#dbeafe'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06',
                                   facecolor=color, edgecolor='#64748b', linewidth=1.2))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=11)

    def arrow(ax, start, end, label=None):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops={'arrowstyle': '-|>', 'color': '#475569', 'lw': 1.5})
        if label:
            ax.text((start[0]+end[0])/2+.08, (start[1]+end[1])/2+.1,
                    label, fontsize=9, color='#475569')

    def save(fig, name):
        fig.tight_layout()
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(args.output_dir / f'{name}.{ext}', dpi=180, facecolor='white')
        plt.close(fig)

    fig, ax = canvas('N-body experiment workflow', 'One physical model; four execution paths; measured results and saved particle states')
    box(ax, .3, 3.1, 2.2, 1.4, 'Particle dataset\nm, x, y, z, vx, vy, vz')
    for y, label in [(5, 'Sequential'), (3.8, 'OpenMP'), (2.6, 'MPI'), (1.4, 'CUDA: naive / tiled')]:
        box(ax, 3.5, y, 2.7, .8, label)
        arrow(ax, (2.5, 3.8), (3.5, y+.4))
        arrow(ax, (6.2, y+.4), (7.2, 3.8))
    box(ax, 7.2, 3.1, 1.8, 1.4, 'Force update\nVelocity update\nPosition update', '#dcfce7')
    box(ax, 9.6, 4.4, 2, 1, 'Timing records\nRepeated runs', '#fef3c7')
    box(ax, 9.6, 2.2, 2, 1, 'Particle states\nCorrectness checks', '#fef3c7')
    arrow(ax, (9, 4), (9.6, 4.9))
    arrow(ax, (9, 3.5), (9.6, 2.7))
    ax.text(.3, .45, 'Analysis: runtime, speedup, efficiency, overhead and tiling benefit.\nSnapshot generation is separate from the timed benchmark study.', color='#475569')
    save(fig, 'workflow')

    fig, ax = canvas('OpenMP: shared positions, disjoint updates', 'Example with four threads and N divisible by four; static scheduling distributes outer-loop particles')
    box(ax, .8, 4.8, 10.4, .8, 'Shared particle array: every thread reads all start-of-step positions', '#dcfce7')
    for i in range(4):
        x = .45 + 2.95*i
        box(ax, x, 2.9, 2.3, 1.1, f'Thread {i}\nOwn particle range\nCompute a; update v')
        arrow(ax, (x+1.15, 4.8), (x+1.15, 4))
        arrow(ax, (x+1.15, 2.9), (x+1.15, 2.3))
        box(ax, x, 1, 2.3, .8, f'Thread {i}\nUpdate own positions')
        arrow(ax, (x+1.15, 2.1), (x+1.15, 1.8))
    box(ax, .45, 2.05, 11.15, .3, 'Implicit barrier: all force / velocity updates complete', '#fef3c7')
    ax.text(.45, .4, 'A second implicit barrier completes the position loop before the next timestep. No particle is updated by two threads.', fontsize=10)
    save(fig, 'openmp_partitioning')

    fig, ax = canvas('MPI: local work, globally replicated states', 'Rank 0 loads data and broadcasts it once. Each timestep uses MPI_Allgatherv to exchange updated particles.')
    for i in range(4):
        x = .45 + 2.95*i
        box(ax, x, 4, 2.3, 1.3, f'Rank {i}\nFull array replica\nUpdate owned range')
        arrow(ax, (x+1.15, 4), (x+1.15, 3.25))
        box(ax, x, 1.25, 2.3, 1, f'Rank {i}\nFull updated array', '#dcfce7')
        arrow(ax, (x+1.15, 2.75), (x+1.15, 2.25))
    box(ax, .45, 2.75, 11.15, .5, 'MPI_Allgatherv (in place): exchange each rank\'s updated Particle records', '#fef3c7')
    ax.text(.45, .55, 'Counts and displacements handle uneven partitions. Repeat force, position and exchange steps for every iteration.', fontsize=10)
    save(fig, 'mpi_exchange')

    fig, ax = canvas('CUDA: reuse a particle tile inside each block', 'One thread owns one target particle. Source particles are processed in increasing index order.')
    box(ax, .5, 4.8, 11, .75, 'Global memory: source particles [0 ... N-1]', '#dcfce7')
    arrow(ax, (6, 4.8), (6, 4.1), 'Cooperative load: one source per thread')
    box(ax, .5, 3.25, 11, .85, 'Shared-memory tile: (x, y, z, mass) for up to blockDim.x sources')
    ax.text(6, 2.9, '__syncthreads(): tile ready', ha='center', color='#92400e')
    for i, label in enumerate(['Thread 0', 'Thread 1', '...', 'Thread B-1']):
        x = .5 + 2.9*i
        box(ax, x, 1.45, 2.3, 1, f'{label}\nAccumulate force\non its target')
        arrow(ax, (x+1.15, 3.25), (x+1.15, 2.45))
    ax.text(.5, .85, 'Barrier before replacing the tile; repeat for all source tiles, then update velocities.', fontsize=11)
    ax.text(.5, .35, 'A separate kernel updates positions. Naive baseline reads sources directly from global memory without a shared tile.', fontsize=10)
    save(fig, 'cuda_tiling')
    print(f'Saved four diagrams as PNG, PDF and SVG in {args.output_dir}')


if __name__ == '__main__':
    main()
