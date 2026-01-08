import os
import re
import time

import pandas as pd
from tqdm.auto import tqdm
from upath import UPath
from cellmap_segmentation_challenge import evaluate, TRUTH_PATH

CMD_TEMPLATE = 'bsub -o {out} -e {err} -n {num_cpus} -J eval_{eval_ID} "csc evaluate -s {submission_path} -r {result_file} -t {truth_path}"'


def get_result_file(submission_path, eval_ID):
    submission_path = UPath(submission_path)
    result_file = submission_path.parent / f"eval_{eval_ID}.results"
    return str(result_file)


def _eval_one_local(
    submission_path,
    eval_ID,
    truth_path=TRUTH_PATH,
):
    result_file = get_result_file(submission_path, eval_ID)
    evaluate.score_submission(submission_path, result_file, truth_path)


def _eval_one_cluster(
    submission_path,
    eval_ID,
    truth_path=TRUTH_PATH,
    num_cpus=48,
    cmd_template=CMD_TEMPLATE,
):
    result_file = get_result_file(submission_path, eval_ID)
    out = os.path.abspath(f"{eval_ID}.out")
    err = os.path.abspath(f"{eval_ID}.err")
    # Remove any existing output files
    for f in (out, err, result_file):
        try:
            os.remove(f)
        except FileNotFoundError:
            pass
    cmd = cmd_template.format(
        eval_ID=eval_ID,
        submission_path=submission_path,
        result_file=result_file,
        truth_path=truth_path,
        out=out,
        err=err,
        num_cpus=num_cpus,
    )
    print(f"Submitting job with command:\n{cmd}")
    rc = os.system(cmd)
    if rc != 0:
        raise RuntimeError(f"bsub failed for eval_ID={eval_ID} (exit={rc})")
    return out


def monitor_jobs(job_logs):
    SUCCESS_RE = re.compile(r"^\s*Successfully completed\.\s*$", re.I | re.M)
    EXITCODE_RE = re.compile(r"^\s*Exited with exit code\s+(\d+)\.\s*$", re.I | re.M)
    SIGNAL_RE = re.compile(r"^\s*Exited with signal\s+(\d+)\.\s*$", re.I | re.M)

    def tail_text(path: str, max_bytes: int = 65536) -> str:
        """Read up to the last max_bytes of a text file (best-effort)."""
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - max_bytes), os.SEEK_SET)
                data = f.read()
            return data.decode(errors="replace")
        except FileNotFoundError:
            return ""
        except OSError:
            return ""

    def classify_from_stdout(stdout_path: str):
        t = tail_text(stdout_path)
        if not t:
            return None  # can't tell yet

        if SUCCESS_RE.search(t):
            return ("success", 0)

        m = EXITCODE_RE.search(t)
        if m:
            return ("failed", int(m.group(1)))

        m = SIGNAL_RE.search(t)
        if m:
            return ("failed_signal", int(m.group(1)))

        # footer not present yet / or unusual format
        return None

    unfinished = set(job_logs)  # here jobs should map to stdout_path (see below)
    failed = set()
    status = tqdm(total=len(unfinished), desc="Evaluating...")

    while unfinished:
        for stdout_path in list(unfinished):
            result = classify_from_stdout(stdout_path)

            if result is None:
                continue

            kind, code = result
            if kind == "success":
                status.set_postfix_str(f"{stdout_path}: success")
            elif kind == "failed":
                status.set_postfix_str(f"{stdout_path}: failed (exit {code})")
                failed.add(stdout_path)
            else:
                status.set_postfix_str(f"{stdout_path}: failed (signal {code})")
                failed.add(stdout_path)
            unfinished.remove(stdout_path)
            status.update(1)
        time.sleep(30)

    status.close()
    return failed


def eval_batch(csv_path, cluster=False, cmd_template=CMD_TEMPLATE, num_cpus=48):
    jobs = []
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        submission_path = row["data_path"]
        eval_ID = row["evaluation_id"]

        if cluster:
            job_log = _eval_one_cluster(
                submission_path, eval_ID, cmd_template=cmd_template, num_cpus=num_cpus
            )
            jobs.append(job_log)
        else:
            _eval_one_local(submission_path, eval_ID)

    if cluster:
        # Monitor job statuses
        failed = monitor_jobs(jobs)
        if failed:
            print("Some jobs failed. Review the following logs:")
            for job in failed:
                print(f" - {job}")
            # Print to timestamped file
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            fail_log = f"failed_jobs_{timestamp}.txt"
            with open(fail_log, "w") as f:
                for job in failed:
                    f.write(f"{job}\n")
            print(f"Failed job logs written to: {fail_log}")
        else:
            print("All evaluations completed successfully.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch evaluation of submissions for the CellMap Segmentation Challenge."
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the CSV file containing submission paths and eval IDs.",
    )
    parser.add_argument(
        "--cluster",
        "-c",
        action="store_true",
        help="If set, submit evaluation jobs to a cluster instead of running locally.",
    )
    parser.add_argument(
        "--num_cpus",
        "-n",
        type=int,
        default=48,
        help="Number of CPUs to request per job when submitting to cluster. Default is 48.",
    )

    args = parser.parse_args()
    eval_batch(args.csv_path, cluster=args.cluster, num_cpus=args.num_cpus)
