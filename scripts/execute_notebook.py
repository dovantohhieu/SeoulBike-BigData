"""Chạy notebook từ kernel sạch và lưu đầu ra cùng nhật ký kiểm tra."""
from pathlib import Path
import argparse
import os
import time
import json
import sys
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'notebooks/Demo_BigData.ipynb')
    parser.add_argument('--skip-training',action='store_true')
    parser.add_argument('--skip-benchmark',action='store_true')
    parser.add_argument('--timeout',type=int,default=1200)
    parser.add_argument('--in-process',action='store_true',help='Dùng IPython kernel trong tiến trình mới khi môi trường chặn socket Jupyter')
    args=parser.parse_args()
    notebook=nbformat.read(ROOT/'notebooks/Demo_BigData.ipynb',as_version=4)
    if args.skip_training:
        notebook.cells[1].source=notebook.cells[1].source.replace('RETRAIN_MODELS = True','RETRAIN_MODELS = False')
    if args.skip_benchmark:
        notebook.cells[1].source=notebook.cells[1].source.replace('MEASURE_BENCHMARK = True','MEASURE_BENCHMARK = False')
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='2'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    def progress(cell_index,**kwargs):
        print(f'Cell {cell_index+1}/{len(notebook.cells)}',flush=True)
    try:
        if args.in_process:
            from ipykernel.inprocess.manager import InProcessKernelManager
            from IPython.utils.capture import capture_output
            manager=InProcessKernelManager()
            manager.start_kernel()
            previous=Path.cwd()
            os.chdir(ROOT)
            try:
                for i,cell in enumerate(notebook.cells):
                    progress(cell_index=i)
                    if cell.cell_type!='code':continue
                    with capture_output(stdout=True,stderr=True,display=True) as captured:
                        result=manager.kernel.shell.run_cell(cell.source,store_history=True)
                    outputs=[]
                    if captured.stdout:outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
                    if captured.stderr:outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
                    for display in captured.outputs:
                        outputs.append(nbformat.v4.new_output('display_data',data=display.data,metadata=display.metadata))
                    cell.execution_count=result.execution_count
                    cell.outputs=outputs
                    error=result.error_before_exec or result.error_in_exec
                    if error:
                        cell.outputs.append(nbformat.v4.new_output('error',ename=type(error).__name__,evalue=str(error),traceback=[str(error)]))
                    nbformat.write(notebook,args.output)
                    if error:raise error
            finally:
                os.chdir(previous)
                manager.shutdown_kernel()
        else:
            client=NotebookClient(notebook,timeout=args.timeout,kernel_name='python3',
                resources={'metadata':{'path':str(ROOT)}},allow_errors=False,on_cell_start=progress)
            client.execute()
    finally:
        nbformat.write(notebook,args.output)
    code=[c for c in notebook.cells if c.cell_type=='code']
    errors=[o for c in code for o in c.get('outputs',[]) if o.output_type=='error']
    if errors:raise RuntimeError(f'Notebook còn {len(errors)} ô lỗi.')
    result={'status':'passed','cells':len(notebook.cells),'code_cells':len(code),
            'executed_code_cells':sum(c.execution_count is not None for c in code),
            'errors':len(errors),'elapsed_seconds':round(time.perf_counter()-start,2),
            'training_requested':not args.skip_training,'benchmark_requested':not args.skip_benchmark,
            'execution_method':'IPython in-process kernel in fresh Python process' if args.in_process else 'nbclient Jupyter TCP kernel'}
    output=args.output.with_suffix('.execution.json')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
