import pandas as pd
import matplotlib.pyplot as plt
import os

def milestone3_analysis():
    datasets = ['fb', 'osmc', 'books']

    # Map keys → pretty labels (insert % first)
    workloads = {
        '10pct_insert': '10% insert (10% insert, 90% lookup)',
        '90pct_insert': '90% insert (90% insert, 10% lookup)',
    }
    # These suffixes must match your generated CSV filenames
    suffixes = {
        '10pct_insert': '0.100000i_0m_mix',
        '90pct_insert': '0.900000i_0m_mix',
    }

    # Raw index names → nicer tick labels
    indexes = ['DynamicPGM', 'LIPP', 'HybridPGMLIPP']

    os.makedirs('analysis_results', exist_ok=True)

    for key, pretty in workloads.items():
        suffix = suffixes[key]

        fig, axs = plt.subplots(2, len(datasets),
                                figsize=(4*len(datasets), 8),
                                squeeze=False)

        for col, ds in enumerate(datasets):
            fname = (
                f"results/{ds}_100M_public_uint64_"
                f"ops_2M_0.000000rq_0.500000nl_{suffix}_results_table.csv"
            )
            df = pd.read_csv(fname)

            # Gather numbers
            throughput = []
            size_bytes = []
            for idx in indexes:
                sub = df[df['index_name'] == idx]
                if not sub.empty:
                    # average the three mixed‐throughput runs
                    tp_cols = [c for c in sub.columns if 'mixed_throughput_mops' in c]
                    avg_tp = sub[tp_cols].mean(axis=1).max()
                    throughput.append(avg_tp)
                    size_bytes.append(sub['index_size_bytes'].iloc[0])
                else:
                    throughput.append(0)
                    size_bytes.append(0)

            # Top: throughput
            ax = axs[0, col]
            ax.bar(indexes, throughput)
            ax.set_title(f"{ds} — Throughput")
            ax.set_ylabel("Throughput (Mops/s)")
            ax.set_ylim(0, max(throughput)*1.2 if any(throughput) else 1)

            # Bottom: index size
            ax = axs[1, col]
            ax.bar(indexes, size_bytes)
            ax.set_title(f"{ds} — Index size")
            ax.set_ylabel("Size (bytes)")
            ax.set_ylim(0, max(size_bytes)*1.2 if any(size_bytes) else 1)

        fig.suptitle(f"Mixed Workload: {pretty}", fontsize=16)
        plt.tight_layout(rect=[0,0,1,0.95])

        out_png = f"analysis_results/{key}.png"
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        print(f"Saved plot: {out_png}")

if __name__ == "__main__":
    milestone3_analysis()