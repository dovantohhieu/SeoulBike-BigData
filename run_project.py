"""Chạy từ đầu đến cuối bằng các module độc lập, dừng ngay khi một bước lỗi."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description="Seoul Bike Big Data project")
    parser.add_argument("--all",action="store_true",help="Kiểm thử, train sklearn/Spark, benchmark và dựng biểu đồ/demo offline")
    parser.add_argument("--train",action="store_true")
    parser.add_argument("--benchmark",action="store_true")
    parser.add_argument("--analyze",action="store_true")
    parser.add_argument("--demo",action="store_true")
    parser.add_argument("--lab",action="store_true",help="Chạy demo Spark ETL, Parquet phân vùng và SQL")
    parser.add_argument("--rows",type=int,choices=[8760,100000,1000000],default=8760)
    args=parser.parse_args()
    env={**os.environ,"OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2","SPARK_LOCAL_IP":"127.0.0.1","TZ":"UTC"}
    def call(*command):
        print("Running:"," ".join(command),flush=True)
        subprocess.run([sys.executable,*command],cwd=ROOT,env=env,check=True)
    if args.all:
        call("-m","unittest","discover","-s","tests","-p","test_core.py","-v")
    if args.all or args.train:
        call("-m","src.train")
        call("-m","src.spark_pipeline")
    if args.all or args.benchmark:
        call("-m","src.benchmark")
    if args.all or args.lab:
        call("-m","src.bigdata_lab","--rows",str(args.rows))
    if args.all or args.analyze:
        call("-m","src.analyze")
    if args.demo:
        call("-m","streamlit","run","app.py")
    if not any([args.all,args.train,args.benchmark,args.analyze,args.demo,args.lab]):parser.print_help()

if __name__=="__main__":main()
