#!/usr/bin/env python3
"""Analyze legacy or labelled benchmark CSVs without pooling distinct experiments."""
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STYLE = {'seq': ('Sequential', 'black', 'o'), 'omp': ('OpenMP', '#2563eb', 's'),
         'mpi': ('MPI', '#059669', '^'), 'cuda': ('CUDA tiled', '#dc2626', 'D'),
         'cuda_naive': ('CUDA naive', '#9333ea', 'v')}
KEYS = ['impl', 'n', 'iters', 'units']


def summarize(data):
    return data.groupby(KEYS, as_index=False).agg(
        total_s=('total_s', 'median'), lo=('total_s', 'min'), hi=('total_s', 'max'),
        aux_s=('aux_s', 'median'), aux_lo=('aux_s', 'min'), aux_hi=('aux_s', 'max'),
        comm_pct=('comm_pct', 'median'), runs=('total_s', 'count'))


def ratios(summary):
    reference = summary[summary.impl == 'seq'][['n', 'iters', 'total_s', 'lo', 'hi']]
    reference = reference.rename(columns={'total_s': 'baseline', 'lo': 'base_lo', 'hi': 'base_hi'})
    result = summary.merge(reference, on=['n', 'iters'], how='left', validate='many_to_one')
    result['speedup'] = result.baseline / result.total_s
    result['speed_lo'] = result.base_lo / result.hi
    result['speed_hi'] = result.base_hi / result.lo
    result['efficiency'] = result.speedup / result.units
    result.loc[~result.impl.isin(['omp', 'mpi']), 'efficiency'] = np.nan
    result['karp_flatt'] = np.nan
    mask = result.impl.isin(['omp', 'mpi']) & (result.units > 1)
    result.loc[mask, 'karp_flatt'] = ((1 / result.loc[mask, 'speedup'] - 1 / result.loc[mask, 'units']) /
                                           (1 - 1 / result.loc[mask, 'units']))
    return result


def fixed_units(summary):
    parts = []
    for _, group in summary.groupby('impl'):
        units = group.groupby('units').n.nunique().idxmax()
        parts.append(group[group.units == units])
    return pd.concat(parts, ignore_index=True) if parts else summary.copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('results/results.csv'))
    parser.add_argument('--output-dir', type=Path, help='Default: <input directory>/analysis')
    parser.add_argument('--iters', type=int, help='Select one iteration count if the CSV contains several')
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f'Missing input: {args.input}')
    df = pd.read_csv(args.input)
    required = set(KEYS + ['total_s', 'avg_iter_s', 'aux_s'])
    if not required.issubset(df.columns):
        parser.error(f'Missing columns: {sorted(required - set(df.columns))}')
    if df.empty:
        parser.error('No measurements in input')
    for column in required - {'impl'}:
        df[column] = pd.to_numeric(df[column], errors='raise')
        if not np.isfinite(df[column]).all():
            parser.error(f'Non-finite {column}')
    if (df[['n', 'iters', 'units', 'total_s', 'avg_iter_s']] <= 0).any().any() or (df.aux_s < 0).any():
        parser.error('Counts and timings must be positive; aux_s must be nonnegative')
    if (df[['n', 'iters', 'units']] % 1 != 0).any().any():
        parser.error('Particle, iteration and worker counts must be integers')
    if not set(df.impl).issubset(STYLE):
        parser.error(f'Unknown implementation labels: {set(df.impl) - set(STYLE)}')
    if (df.loc[df.impl == 'seq', 'units'] != 1).any():
        parser.error('Sequential baseline must have units=1')
    if args.iters is not None:
        df = df[df.iters == args.iters].copy()
    if df.empty or df.iters.nunique() != 1:
        parser.error('Select a single available iteration count with --iters')
    legacy = 'experiment' not in df.columns
    if not legacy:
        if df.experiment.isna().any() or not set(df.experiment).issubset({'size', 'scaling', 'block', 'cuda_compare'}):
            parser.error('Invalid experiment labels')
        if 'rep' not in df or df.duplicated(['experiment', 'rep', *KEYS]).any():
            parser.error('Missing repetition IDs or duplicated measurement IDs')
        repetition = pd.to_numeric(df.rep, errors='coerce')
        if repetition.isna().any() or (repetition <= 0).any() or (repetition % 1 != 0).any():
            parser.error('Repetition IDs must be positive integers')
    output = args.output_dir or args.input.parent / 'analysis'
    output.mkdir(parents=True, exist_ok=False)
    figure_dir = output / 'figures'
    figure_dir.mkdir()
    notes = ['Input: ' + str(args.input.resolve()), f'Iterations per run: {int(df.iters.iloc[0])}',
             'Runtime bars: median and observed min-max range; not confidence intervals.',
             'Speedup bars: baseline_min/parallel_max to baseline_max/parallel_min; not paired confidence intervals.',
             'CUDA total_s measures the device loop, excluding setup, warm-up and transfers.',
             'CUDA aux_s measures initial H2D + final D2H only; kernel + transfer is not full application time.',
             'MPI aux_s is the maximum accumulated collective time across ranks, including waiting.',
             'MPI collective time and rank-0 total are not an additive phase decomposition.']
    if legacy:
        notes.append('Legacy CSV: experiment/repetition IDs unavailable. Existing duplicates are pooled by configuration; fixed units inferred by size coverage.')
    if (df.total_s < 0.01).any():
        notes.append('Some runs are shorter than 10 ms. Longer runs and more repetitions are recommended.')
    df['comm_pct'] = 100 * df.aux_s / df.total_s
    count = 0
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

    def data(experiment):
        return df if legacy else df[df.experiment == experiment]

    def save(fig, name, note='Medians; bars show observed variability'):
        nonlocal count
        fig.text(.5, .01, note, ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .05, 1, 1))
        for suffix in ['png', 'pdf']:
            fig.savefig(figure_dir / f'{name}.{suffix}', dpi=180)
        plt.close(fig)
        count += 1

    def table(frame, name):
        frame.to_csv(output / f'{name}.csv', index=False)
        # Plain tabular output avoids pandas Styler's optional Jinja2 dependency.
        def cell(value):
            if pd.isna(value): return '--'
            text = f'{value:.5g}' if isinstance(value, (float, np.floating)) else str(value)
            return text.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')
        lines = [r'\begin{tabular}{' + 'l' * len(frame.columns) + '}', r'\hline',
                 ' & '.join(map(cell, frame.columns)) + r' \\', r'\hline']
        lines.extend(' & '.join(map(cell, row)) + r' \\' for row in frame.itertuples(index=False, name=None))
        lines.extend([r'\hline', r'\end{tabular}'])
        (output / f'{name}.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    def line(ax, group, x, y='total_s', low='lo', high='hi', label=None):
        group = group.sort_values(x)
        group = group.dropna(subset=[y, low, high])
        if group.empty:
            return
        impl = group.impl.iloc[0]
        title, color, marker = STYLE[impl]
        errors = np.maximum(0, np.array([group[y] - group[low], group[high] - group[y]]))
        ax.errorbar(group[x], group[y], yerr=errors, marker=marker, color=color,
                    capsize=3, linewidth=1.4, label=label or title)

    def decorate(ax, title, xlabel, ylabel, logx=False, logy=False):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        if logx: ax.set_xscale('log')
        if logy: ax.set_yscale('log')
        ax.grid(True, alpha=.25, which='both')
        handles, _ = ax.get_legend_handles_labels()
        if handles: ax.legend(fontsize=8)

    size = ratios(fixed_units(summarize(data('size'))))
    table(size, 'summary_size')
    if not size.empty:
        for name, y, low, high, ylabel in [('time_vs_n', 'total_s', 'lo', 'hi', 'Timed simulation loop (s)'),
                                          ('speedup_vs_n', 'speedup', 'speed_lo', 'speed_hi', 'Speedup over sequential')]:
            fig, ax = plt.subplots(figsize=(7, 4.8))
            for impl, group in size.groupby('impl'):
                if y == 'speedup' and impl == 'seq': continue
                line(ax, group, 'n', y, low, high, f'{STYLE[impl][0]} (units={int(group.units.iloc[0])})')
            decorate(ax, 'Runtime vs particle count' if y == 'total_s' else 'Speedup vs particle count',
                     'Particles N', ylabel, True, y == 'total_s')
            save(fig, name)
        fig, ax = plt.subplots(figsize=(7, 4.8))
        fits = []
        for impl, group in size.groupby('impl'):
            group = group.sort_values('n')
            if len(group) < 3: continue
            slope, intercept = np.polyfit(np.log(group.n), np.log(group.total_s), 1)
            line(ax, group, 'n', label=f'{STYLE[impl][0]}: slope {slope:.2f}')
            ax.plot(group.n, np.exp(intercept) * group.n ** slope, '--', color=STYLE[impl][1], alpha=.5)
            fits.append(f'{impl}: slope={slope:.4f}; N={int(group.n.min())}..{int(group.n.max())}')
        if len(size):
            ref = size[size.impl == 'seq'].sort_values('n')
            if ref.empty: ref = size.sort_values('n')
            x = np.sort(size.n.unique())
            ax.plot(x, ref.total_s.iloc[0] * (x / ref.n.iloc[0]) ** 2, ':', color='gray', label='N squared reference')
        decorate(ax, 'Measured scaling and fitted slopes', 'Particles N', 'Timed simulation loop (s)', True, True)
        save(fig, 'complexity_loglog', 'Observed slopes describe this workload range, not a change in algorithmic complexity')
        (output / 'complexity_fit.txt').write_text('\n'.join(fits) + '\n', encoding='utf-8')

    scaling = ratios(summarize(data('scaling')))
    scaling = scaling[scaling.impl.isin(['omp', 'mpi'])].copy()
    eligible = scaling.groupby('n').units.nunique()
    scaling = scaling[scaling.n.isin(eligible[eligible > 1].index)]
    table(scaling, 'summary_scaling')
    if not scaling.empty:
        sizes = sorted(scaling.n.unique())
        for efficiency in [False, True]:
            fig, axes = plt.subplots(1, len(sizes), figsize=(5 * len(sizes), 4.5), squeeze=False)
            for ax, n in zip(axes.flat, sizes):
                group = scaling[scaling.n == n].copy()
                if efficiency:
                    for column in ['speedup', 'speed_lo', 'speed_hi']: group[column] /= group.units
                for _, sub in group.groupby('impl'): line(ax, sub, 'units', 'speedup', 'speed_lo', 'speed_hi')
                x = sorted(group.units.unique())
                ax.plot(x, np.ones(len(x)) if efficiency else x, '--', color='gray', label='Ideal')
                ax.set_xticks(x)
                decorate(ax, f'N = {n:,}', 'Threads / processes', 'Efficiency' if efficiency else 'Speedup')
            save(fig, 'strong_scaling_efficiency' if efficiency else 'strong_scaling_speedup')
    else: notes.append('SKIP: strong scaling requires multiple worker counts.')

    mpi = scaling[scaling.impl == 'mpi']
    if not mpi.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for n, group in mpi.groupby('n'):
            group = group.sort_values('units')
            axes[0].errorbar(group.units, group.aux_s, yerr=[group.aux_s-group.aux_lo, group.aux_hi-group.aux_s], marker='o', capsize=3, label=f'N={n:,}')
            axes[1].plot(group.units, group.comm_pct, 'o-', label=f'N={n:,}')
        decorate(axes[0], 'MPI collective time', 'Processes', 'Maximum accumulated collective time (s)')
        decorate(axes[1], 'MPI collective-time ratio', 'Processes', 'Collective / total time (%)')
        save(fig, 'mpi_communication', 'Collectives include waiting; maximum across ranks is not a synchronized phase breakdown')

    cuda = size[size.impl == 'cuda']
    if not cuda.empty:
        fig, ax = plt.subplots(figsize=(7, 4.8))
        for y, lo, hi, label, color in [('total_s', 'lo', 'hi', 'Device loop', '#dc2626'), ('aux_s', 'aux_lo', 'aux_hi', 'H2D + D2H', '#2563eb')]:
            group = cuda.sort_values('n')
            ax.errorbar(group.n, group[y], yerr=[group[y]-group[lo], group[hi]-group[y]], marker='o', capsize=3, label=label, color=color)
        decorate(ax, 'CUDA computation and transfer costs', 'Particles N', 'Time (s)', True)
        save(fig, 'cuda_transfer_cost', 'Transfers exclude warm-up reset; these measurements do not include full application startup')

    blocks = summarize(data('block'))
    blocks = blocks[blocks.impl == 'cuda']
    eligible = blocks.groupby('n').units.nunique()
    blocks = blocks[blocks.n.isin(eligible[eligible > 1].index)]
    table(blocks, 'summary_cuda_block')
    if not blocks.empty:
        fig, ax = plt.subplots(figsize=(7, 4.8))
        for n, group in blocks.groupby('n'):
            group = group.sort_values('units')
            ax.errorbar(group.units, group.total_s, yerr=[group.total_s-group.lo, group.hi-group.total_s], marker='D', capsize=3, label=f'N={n:,}')
        ax.set_xscale('log', base=2)
        decorate(ax, 'CUDA block-size sensitivity', 'Threads per block', 'Device loop time (s)')
        save(fig, 'cuda_block_size')

    comparison = summarize(data('cuda_compare'))
    naive = comparison[comparison.impl == 'cuda_naive']
    tiled = comparison[comparison.impl == 'cuda']
    paired = naive.merge(tiled, on=['n', 'iters', 'units'], suffixes=('_naive', '_tiled'))
    if not paired.empty:
        paired['speedup'] = paired.total_s_naive / paired.total_s_tiled
        paired['speed_lo'] = paired.lo_naive / paired.hi_tiled
        paired['speed_hi'] = paired.hi_naive / paired.lo_tiled
        table(paired, 'summary_cuda_comparison')
        units = sorted(paired.units.unique())
        fig, axes = plt.subplots(1, len(units), figsize=(6 * len(units), 4.5), squeeze=False)
        fig2, axes2 = plt.subplots(1, len(units), figsize=(6 * len(units), 4.5), squeeze=False)
        for ax, ax2, u in zip(axes.flat, axes2.flat, units):
            matched = paired[paired.units == u].sort_values('n')
            for impl in ['cuda_naive', 'cuda']:
                group = comparison[(comparison.units == u) & (comparison.impl == impl) & comparison.n.isin(matched.n)]
                line(ax, group, 'n')
            decorate(ax, f'CUDA kernels: block={u}', 'Particles N', 'Device loop time (s)', True, True)
            ax2.errorbar(matched.n, matched.speedup, yerr=[matched.speedup-matched.speed_lo, matched.speed_hi-matched.speedup], marker='o', capsize=3, label='Naive / tiled')
            ax2.axhline(1, color='gray', linestyle='--')
            decorate(ax2, f'Tiling benefit: block={u}', 'Particles N', 'Speedup over naive', True)
        save(fig, 'cuda_naive_vs_tiled')
        save(fig2, 'cuda_tiling_speedup')
    else: notes.append('SKIP: naive-versus-tiled plots require measured cuda_compare results for both kernels.')
    notes.append(f'Generated {count} figures (PNG and PDF).')
    (output / 'analysis_notes.txt').write_text('\n'.join(notes) + '\n', encoding='utf-8')
    print('\n'.join(notes))
    print('Output:', output)


if __name__ == '__main__':
    main()
