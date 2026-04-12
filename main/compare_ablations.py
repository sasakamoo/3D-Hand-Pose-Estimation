"""
compare_ablations.py
====================
Reads ablation log files and prints a summary table + plots learning curves.

Usage (from HOISDF-main/ or anywhere):
    python main/compare_ablations.py
    python main/compare_ablations.py --log_root /path/to/outputs/model_dump
"""

import os, sys, argparse, glob, re
import numpy as np

_file_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_file_dir)


def parse_log(log_path):
    """
    Parse an ablation_log.txt and return:
        epochs   : list of int
        losses   : list of float
        mje_mm   : list of float
    """
    epochs, losses, mjes = [], [], []
    pat = re.compile(
        r'Epoch\s+(\d+)\s+loss=([\d.]+)\s+eval_MJE=([\d.]+)mm')
    with open(log_path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                mjes.append(float(m.group(3)))
    return epochs, losses, mjes


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--log_root', type=str,
                   default=os.path.join(_root_dir, 'outputs', 'model_dump'),
                   help='Directory containing abl_* sub-folders')
    p.add_argument('--save_dir', type=str,
                   default=os.path.join(_root_dir, 'outputs', 'ablation_comparison'),
                   help='Where to save comparison plots')
    return p.parse_args()


ABLATION_LABELS = {
    'abl_baseline':    'Baseline (no change)',
    'abl_bone_loss':   '+ Bone length loss',
    'abl_reproj_loss': '+ 2D reprojection loss',
    'abl_lr_sched':    'Cosine LR schedule',
    'abl_sdf_pts':     '2× SDF points (1200)',
}

COLORS = ['#4FC3F7', '#81C784', '#FFB74D', '#F06292', '#CE93D8']


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Find all ablation log files
    results = {}
    for abl_dir in sorted(glob.glob(os.path.join(args.log_root, 'abl_*'))):
        name = os.path.basename(abl_dir)
        log  = os.path.join(abl_dir, 'log', name, 'ablation_log.txt')
        if not os.path.exists(log):
            # Try alternate path
            logs = glob.glob(os.path.join(abl_dir, '**', 'ablation_log.txt'),
                             recursive=True)
            if not logs:
                print(f'  WARNING: no log found for {name}')
                continue
            log = logs[0]

        epochs, losses, mjes = parse_log(log)
        if not mjes:
            print(f'  WARNING: empty log for {name}')
            continue

        results[name] = {
            'epochs': epochs,
            'losses': losses,
            'mjes':   mjes,
            'best':   min(mjes),
            'final':  mjes[-1],
            'label':  ABLATION_LABELS.get(name, name),
        }
        print(f'  {name:25s}  best={min(mjes):.2f}mm  '
              f'final={mjes[-1]:.2f}mm  ({len(mjes)} epochs)')

    if not results:
        print('\nNo ablation results found. Check --log_root path.')
        sys.exit(0)

    # ── Summary table ─────────────────────────────────────────────────────────
    print('\n' + '═'*62)
    print(f'  {"Ablation":<30}  {"Best MJE":>10}  {"Final MJE":>10}')
    print('─'*62)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['best'])
    for name, r in sorted_results:
        marker = ' ★' if r['best'] == min(v['best'] for v in results.values()) else ''
        print(f'  {r["label"]:<30}  {r["best"]:>9.2f}mm  '
              f'{r["final"]:>9.2f}mm{marker}')
    print('═'*62)

    # ── Plots ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                                  facecolor='#0d0d0d')
        fig.suptitle('Ablation Study — HOISDF Hand-Only FreiHAND',
                     color='white', fontsize=13)

        for ax in axes:
            ax.set_facecolor('#1a1a1a')
            ax.tick_params(colors='#aaaaaa')
            ax.spines['bottom'].set_color('#444')
            ax.spines['left'].set_color('#444')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(color='#333', linestyle='--', linewidth=0.5)

        for i, (name, r) in enumerate(sorted(results.items())):
            c = COLORS[i % len(COLORS)]
            axes[0].plot(r['epochs'], r['losses'], color=c,
                         label=r['label'], lw=1.5, alpha=0.9)
            axes[1].plot(r['epochs'], r['mjes'], color=c,
                         label=r['label'], lw=2.0, alpha=0.9)
            # Mark best
            best_idx = np.argmin(r['mjes'])
            axes[1].scatter(r['epochs'][best_idx], r['mjes'][best_idx],
                            color=c, s=80, zorder=5)

        axes[0].set_xlabel('Epoch', color='#aaaaaa')
        axes[0].set_ylabel('Train Loss', color='#aaaaaa')
        axes[0].set_title('Training Loss', color='white')

        axes[1].set_xlabel('Epoch', color='#aaaaaa')
        axes[1].set_ylabel('Eval MJE (mm)', color='#aaaaaa')
        axes[1].set_title('Eval MJE ↓ lower is better', color='white')

        axes[1].legend(facecolor='#222', labelcolor='white',
                       fontsize=8, loc='upper right')

        plt.tight_layout()
        plot_path = os.path.join(args.save_dir, 'ablation_curves.png')
        fig.savefig(plot_path, dpi=130, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f'\nPlot saved → {plot_path}')

        # ── Bar chart of best MJE ──────────────────────────────────────────
        fig2, ax = plt.subplots(figsize=(9, 4), facecolor='#0d0d0d')
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='#aaaaaa')
        for spine in ax.spines.values():
            spine.set_color('#444')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', color='#333', linestyle='--', linewidth=0.5)

        names  = [r['label'] for _, r in sorted_results]
        bests  = [r['best']  for _, r in sorted_results]
        bars   = ax.bar(names, bests,
                        color=[COLORS[i] for i in range(len(sorted_results))],
                        width=0.6, alpha=0.9)

        for bar, val in zip(bars, bests):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02,
                    f'{val:.2f}mm', ha='center', va='bottom',
                    color='white', fontsize=9)

        ax.set_ylabel('Best Eval MJE (mm)', color='#aaaaaa')
        ax.set_title('Best MJE per Ablation  ↓ lower is better',
                     color='white')
        plt.xticks(rotation=20, ha='right', color='#aaaaaa', fontsize=8)
        plt.tight_layout()
        bar_path = os.path.join(args.save_dir, 'ablation_bar.png')
        fig2.savefig(bar_path, dpi=130, bbox_inches='tight',
                     facecolor=fig2.get_facecolor())
        plt.close(fig2)
        print(f'Bar chart  → {bar_path}')

    except ImportError:
        print('matplotlib not available — skipping plots')

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.save_dir, 'ablation_results.csv')
    with open(csv_path, 'w') as f:
        f.write('ablation,label,best_mje_mm,final_mje_mm,n_epochs\n')
        for name, r in sorted_results:
            f.write(f'{name},{r["label"]},{r["best"]:.3f},'
                    f'{r["final"]:.3f},{len(r["mjes"])}\n')
    print(f'CSV        → {csv_path}')


if __name__ == '__main__':
    main()