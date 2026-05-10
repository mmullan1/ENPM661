
# ============== Imports ==================
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import contextlib
import io


# import function to move robot in ROS2
from rrt_builder import start_rrt, compute_total_cost
# -------------------------------------------------------------------------------------------------
def benchmark_rrt_algorithms(
    q_start,
    q_goal,
    max_step_size,
    scenes,
    algorithms=None,
    runs_per_case=20,
    output_type="2",
    csv_filename="rrt_benchmark_results.csv",
    summary_filename="rrt_benchmark_summary.csv",
    figure_filename="rrt_benchmark_summary.png",
    quiet=True
):
    """
    Runs each RRT algorithm multiple times in multiple environments and tabulates
    the average cost/time along with spread information.

    Inputs:
        - q_start: starting joint configuration
        - q_goal: goal joint configuration
        - max_step_size: maximum joint-space step size
        - scenes: list of YAML scene files to test
        - algorithms: list of algorithms to test
        - runs_per_case: number of times to run each algorithm in each scene
        - output_type: output mode passed into start_rrt()
            "2" is recommended because it returns the smoothed path
        - csv_filename: raw per-run output file
        - summary_filename: summarized statistics output file
        - figure_filename: saved figure table filename
        - quiet: if True, suppresses print output from each individual RRT run

    Returns:
        - summary_rows: list of dictionaries containing summary statistics
    """

    if algorithms is None:
        algorithms = [
            "rrt",
            "rrt_star",
            "birrt",
            "birrt_star",
            "improved_birrt_star"
        ]

    raw_rows = []
    summary_rows = []

    total_runs = len(scenes) * len(algorithms) * runs_per_case

    print("\n========== RRT Benchmark Started ==========")
    print(f"Scenes: {scenes}")
    print(f"Algorithms: {algorithms}")
    print(f"Runs per algorithm per scene: {runs_per_case}")
    print(f"Total runs: {total_runs}")
    print("===========================================\n")

    for scene in scenes:
        for alg in algorithms:

            print(f"\n--- Running {alg} in {scene} ---")

            costs = []
            times = []
            path_lengths = []
            success_flags = []

            for run_idx in range(runs_per_case):

                t0 = time.perf_counter()

                try:
                    if quiet:
                        with contextlib.redirect_stdout(io.StringIO()):
                            path = start_rrt(
                                q_start,
                                q_goal,
                                max_step_size,
                                scene,
                                output_type,
                                alg
                            )
                    else:
                        path = start_rrt(
                            q_start,
                            q_goal,
                            max_step_size,
                            scene,
                            output_type,
                            alg
                        )

                    t1 = time.perf_counter()
                    elapsed_time = t1 - t0

                    if path is None:
                        success = False
                        cost = np.nan
                        path_length = 0

                    else:
                        success = True
                        cost = compute_total_cost(path)
                        path_length = len(path)

                except Exception as e:
                    t1 = time.perf_counter()
                    elapsed_time = t1 - t0

                    success = False
                    cost = np.nan
                    path_length = 0

                    print(f"Run failed: scene={scene}, alg={alg}, run={run_idx + 1}")
                    print(f"Error: {e}")

                costs.append(cost)
                times.append(elapsed_time)
                path_lengths.append(path_length)
                success_flags.append(success)

                raw_rows.append({
                    "scene": scene,
                    "algorithm": alg,
                    "run": run_idx + 1,
                    "success": success,
                    "cost": cost,
                    "time_sec": elapsed_time,
                    "path_nodes": path_length
                })

                if (run_idx + 1) % 25 == 0:
                    print(f"  Completed {run_idx + 1}/{runs_per_case} runs")

            costs = np.asarray(costs, dtype=float)
            times = np.asarray(times, dtype=float)
            path_lengths = np.asarray(path_lengths, dtype=float)
            success_flags = np.asarray(success_flags, dtype=bool)

            successful_costs = costs[success_flags]
            successful_times = times[success_flags]
            successful_path_lengths = path_lengths[success_flags]

            num_success = int(np.sum(success_flags))
            num_fail = runs_per_case - num_success
            success_rate = num_success / runs_per_case

            def safe_stats(data):
                """
                Computes spread statistics safely.
                Returns NaN values if no successful data exists.
                """

                data = np.asarray(data, dtype=float)

                if len(data) == 0:
                    return {
                        "mean": np.nan,
                        "std": np.nan,
                        "min": np.nan,
                        "q1": np.nan,
                        "median": np.nan,
                        "q3": np.nan,
                        "max": np.nan
                    }

                return {
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data, ddof=1)) if len(data) > 1 else 0.0,
                    "min": float(np.min(data)),
                    "q1": float(np.percentile(data, 25)),
                    "median": float(np.median(data)),
                    "q3": float(np.percentile(data, 75)),
                    "max": float(np.max(data))
                }

            cost_stats = safe_stats(successful_costs)
            time_stats = safe_stats(successful_times)
            node_stats = safe_stats(successful_path_lengths)

            summary_row = {
                "scene": scene,
                "algorithm": alg,
                "runs": runs_per_case,
                "successes": num_success,
                "failures": num_fail,
                "success_rate": success_rate,

                "cost_mean": cost_stats["mean"],
                "cost_std": cost_stats["std"],
                "cost_min": cost_stats["min"],
                "cost_q1": cost_stats["q1"],
                "cost_median": cost_stats["median"],
                "cost_q3": cost_stats["q3"],
                "cost_max": cost_stats["max"],

                "time_mean_sec": time_stats["mean"],
                "time_std_sec": time_stats["std"],
                "time_min_sec": time_stats["min"],
                "time_q1_sec": time_stats["q1"],
                "time_median_sec": time_stats["median"],
                "time_q3_sec": time_stats["q3"],
                "time_max_sec": time_stats["max"],

                "path_nodes_mean": node_stats["mean"],
                "path_nodes_std": node_stats["std"],
                "path_nodes_min": node_stats["min"],
                "path_nodes_q1": node_stats["q1"],
                "path_nodes_median": node_stats["median"],
                "path_nodes_q3": node_stats["q3"],
                "path_nodes_max": node_stats["max"]
            }

            summary_rows.append(summary_row)

            print(f"Finished {alg} in {scene}")
            print(f"  Success rate: {success_rate * 100:.1f}%")
            print(f"  Avg cost:     {cost_stats['mean']:.3f}")
            print(f"  Cost std:     {cost_stats['std']:.3f}")
            print(f"  Avg time:     {time_stats['mean']:.3f} sec")
            print(f"  Time std:     {time_stats['std']:.3f} sec")

    # Save raw per-run results
    raw_fieldnames = [
        "scene",
        "algorithm",
        "run",
        "success",
        "cost",
        "time_sec",
        "path_nodes"
    ]

    with open(csv_filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fieldnames)
        writer.writeheader()
        writer.writerows(raw_rows)

    # Save summary table
    summary_fieldnames = list(summary_rows[0].keys()) if summary_rows else []

    with open(summary_filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n========== RRT Benchmark Complete ==========")
    print(f"Raw results saved to:     {csv_filename}")
    print(f"Summary table saved to:   {summary_filename}")
    print(f"Summary figure saved to:  {figure_filename}")
    print("============================================\n")

    plot_benchmark_summary(
        summary_rows,
        save_filename=figure_filename
    )

    return summary_rows


# -------------------------------------------------------------------------------------------------
def plot_benchmark_summary(summary_rows, save_filename="rrt_benchmark_summary.png"):
    """
    Creates a MatPlotLib figure containing the benchmark summary table.

    Inputs:
        - summary_rows: output from benchmark_rrt_algorithms()
        - save_filename: image filename for saving the table figure

    Returns:
        - None
    """

    if not summary_rows:
        print("No benchmark summary rows to plot.")
        return

    columns = [
        "Scene",
        "Algorithm",
        "Success [%]",
        "Cost Mean",
        "Cost Std",
        "Time Mean [s]",
        "Time Std [s]",
        "Nodes Mean",
        "Nodes Std"
    ]

    table_data = []

    for row in summary_rows:
        table_data.append([
            row["scene"],
            row["algorithm"],
            f"{100.0 * row['success_rate']:.1f}",
            f"{row['cost_mean']:.3f}",
            f"{row['cost_std']:.3f}",
            f"{row['time_mean_sec']:.3f}",
            f"{row['time_std_sec']:.3f}",
            f"{row['path_nodes_mean']:.2f}",
            f"{row['path_nodes_std']:.2f}"
        ])

    fig_height = max(3.0, 0.45 * len(table_data) + 1.5)
    fig_width = 15.0

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    for col_idx in range(len(columns)):
        table[(0, col_idx)].set_text_props(weight="bold")

    ax.set_title(
        "RRT Benchmark Summary",
        fontsize=14,
        fontweight="bold",
        pad=20
    )

    plt.tight_layout()

    if save_filename is not None:
        plt.savefig(save_filename, dpi=300, bbox_inches="tight")
        print(f"Benchmark summary figure saved to: {save_filename}")

    plt.show()

# if __name__ == "__main__":
#     benchmark_rrt_algorithms()