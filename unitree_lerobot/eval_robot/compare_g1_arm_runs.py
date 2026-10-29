"""Compare an arm recording and its offline simulated replay; never opens DDS."""

import argparse
from arm_comparison import load_trial, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", required=True, help="trial.npz, run directory, or DDS JSONL")
    parser.add_argument("--sim", required=True, help="Simulated trial.npz")
    parser.add_argument("--output", required=True, help="New report directory")
    parser.add_argument("--command-topic", choices=("rt/lowcmd", "rt/arm_sdk"))
    parser.add_argument("--rate", type=float, default=100.)
    parser.add_argument("--max-gap", type=float, default=.1)
    args = parser.parse_args()
    summary = write_report(load_trial(args.real, args.command_topic), load_trial(args.sim),
                           args.output, args.rate, args.max_gap)
    for row in summary["joints"]:
        print(f"{row['joint']:24s} RMSE={row['gap_rmse_deg']:.4f} deg")
    print(f"Report: {args.output}; excluded samples: {summary['excluded_samples']}")


if __name__ == "__main__":
    main()
