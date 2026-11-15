"""Sinh dữ liệu nhân bản CHỈ để đo ETL; tuyệt đối không dùng để báo cáo chất lượng dự báo."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from .common import ROOT, save_json, environment
from .data import load_data

def run(sizes, repeats):
    folder = ROOT / "data/benchmark"
    folder.mkdir(parents=True,exist_ok=True)
    raw = load_data().reset_index()[["hour", "count", "temperature", "humidity"]]
    records=[]
    # Một số container mount /proc từ namespace cha: os.getpid() khác /proc/self.
    monitor_pid=int(os.readlink("/proc/self")) if Path("/proc/self").exists() else os.getpid()
    controller=psutil.Process(monitor_pid)
    def cgroup_value(name):
        p=Path("/sys/fs/cgroup")/name
        return p.read_text().strip() if p.exists() else None
    meta = {**environment(), "logical_cpus":psutil.cpu_count(), "physical_cpus":psutil.cpu_count(logical=False),
        "available_cpu_affinity":sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None,
        "cgroup_cpu_max":cgroup_value("cpu.max"), "cgroup_memory_max_bytes":cgroup_value("memory.max"),
        "total_ram_gib":psutil.virtual_memory().total/1024**3, "sizes":sizes,"repeats":repeats,
        "spark":"local[2], 1 GiB driver heap, 2 shuffle partitions", "memory_metric":"maximum sampled sum of parent+descendant RSS; includes JVM, may double-count shared pages",
        "sampling_s":0.025, "cache_policy":"OS cache not flushed; engine order alternates; processes and Spark sessions are fresh each run",
        "workload":"read identical typed CSV, filter valid numeric rows, group by hour, count/sum/mean, collect 24 rows",
        "scope":"single-host local Spark ETL. Replication adds volume, not independent statistical information."}
    save_json(ROOT / "results/benchmark_environment.json",meta)
    for size in sizes:
        frame = raw.iloc[np.arange(size)%len(raw)].reset_index(drop=True)
        frame.insert(0,"row_id",np.arange(size))
        source=folder / f"replicated_{size}.csv"
        frame.to_csv(source,index=False)
        del frame
        reference=None
        for repeat in range(1,repeats+1):
            engines=["pandas","spark"] if repeat%2 else ["spark","pandas"]
            for engine in engines:
                output=ROOT / f"results/bench_{size}_{repeat}_{engine}.json"
                log=ROOT / f"results/bench_{size}_{repeat}_{engine}.log"
                cmd=[sys.executable,"-m","src.benchmark_worker","--engine",engine,"--input",str(source),"--output",str(output)]
                start=time.perf_counter(); peak=0
                with log.open("w") as stream:
                    p=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,"OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2"})
                    while p.poll() is None:
                        rss=0
                        try:
                            processes=controller.children(recursive=True)
                        except psutil.NoSuchProcess:
                            processes=[]
                        for proc in processes:
                            try: rss+=proc.memory_info().rss
                            except (psutil.NoSuchProcess,psutil.AccessDenied): pass
                        peak=max(peak,rss)
                        time.sleep(0.025)
                    if p.returncode:
                        raise RuntimeError(f"Benchmark lỗi, xem {log}")
                result=json.loads(output.read_text())
                agg=pd.DataFrame(result.pop("aggregates")).sort_values("hour").reset_index(drop=True)
                if reference is None: reference=agg
                else:
                    np.testing.assert_allclose(agg.to_numpy(float),reference.to_numpy(float),rtol=1e-10,atol=1e-8)
                assert int(agg["rows"].sum()) == size
                records.append({"size":size,"repeat":repeat,"peak_rss_mib":peak/1024**2,
                    "wall_s":time.perf_counter()-start,"aggregates_match":True,**result})
                pd.DataFrame(records).to_csv(ROOT / "results/benchmark_runs.csv",index=False)
                print(size,repeat,engine,round(result["etl_s"],3),flush=True)
        source.unlink()  # Có thể tái sinh. Không nhét dữ liệu nhân bản vào gói nộp.
    summary=pd.DataFrame(records).groupby(["size","engine"]).agg(
        runs=("etl_s","size"),etl_median_s=("etl_s","median"),etl_min_s=("etl_s","min"),etl_max_s=("etl_s","max"),
        wall_median_s=("wall_s","median"),peak_rss_median_mib=("peak_rss_mib","median")).reset_index()
    summary.to_csv(ROOT / "results/benchmark_summary.csv",index=False)

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--sizes",nargs="+",type=int,default=[8760,100000,1000000])
    p.add_argument("--repeats",type=int,default=3)
    a=p.parse_args()
    if min(a.sizes)<=0 or a.repeats<1: p.error("Quy mô và số lần lặp phải dương.")
    run(a.sizes,a.repeats)
