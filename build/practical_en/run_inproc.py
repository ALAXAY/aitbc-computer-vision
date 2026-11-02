import sys, os, time, nbformat
from nbformat.v4 import new_output
src, dst = sys.argv[1], sys.argv[2]
os.environ["AITBC_DATA"] = os.path.abspath("aitbc_data")
import torch, torchvision          # local Windows env: load torch DLLs before OpenCV 5 (not needed in Colab)
os.makedirs("run_dir", exist_ok=True)
os.chdir("run_dir")
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
shell = InteractiveShell.instance()
import matplotlib
matplotlib.use("module://matplotlib_inline.backend_inline")
nb = nbformat.read(os.path.join("..", src), as_version=4)
t0 = time.time()
n = 0
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    n += 1
    t1 = time.time()
    with capture_output() as cap:
        res = shell.run_cell(cell.source, store_history=True)
    outs = []
    if cap.stdout:
        outs.append(new_output("stream", name="stdout", text=cap.stdout))
    if cap.stderr:
        outs.append(new_output("stream", name="stderr", text=cap.stderr))
    for o in cap.outputs:
        outs.append(new_output("display_data", data=o.data, metadata=o.metadata or {}))
    err = res.error_in_exec or res.error_before_exec
    if err is not None:
        outs.append(new_output("error", ename=type(err).__name__, evalue=str(err), traceback=[]))
    cell.outputs = outs
    cell.execution_count = n
    title = cell.source.splitlines()[0][:70]
    print(f"[{i:02d}] {time.time() - t1:6.1f} s  {title}  {'ERROR ' + type(err).__name__ + ': ' + str(err)[:400] if err else 'ok'}", flush=True)
nbformat.write(nb, os.path.join("..", dst))
print(f"total {time.time() - t0:.0f} s")
