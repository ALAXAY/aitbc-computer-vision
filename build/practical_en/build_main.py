# Assemble the English practical notebook of AIT-BC from the part files.
import os, sys
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ns = {}
for part in ["build_practical_en.py", "part2_topic2.py", "part3_topic3.py", "part4_topic4.py", "part5_topic5.py"]:
    exec(compile(open(part, encoding="utf-8").read(), part, "exec"), ns)
CELLS = ns["CELLS"]


def build(path_out, lib_path="aitbc.py", switches=None):
    lib = open(lib_path, encoding="utf-8").read()
    assert "'''" not in lib
    nb = new_notebook()
    nb.metadata = {"colab": {"provenance": [], "toc_visible": True, "name": os.path.basename(path_out)},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}}
    for kind, text, form in CELLS:
        if kind == "md":
            nb.cells.append(new_markdown_cell(text))
        else:
            src = text.replace("%%LIB%%", lib)
            for old, new in (switches or {}).items():
                src = src.replace(old, new)
            cell = new_code_cell(src)
            if form:
                cell.metadata["cellView"] = "form"
            nb.cells.append(cell)
    nbformat.validate(nb)
    nbformat.write(nb, path_out)
    return nb


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "AITBC_Practical_Course_Colab_EN.ipynb"
    sw = {}
    if len(sys.argv) > 2 and sys.argv[2] == "test":
        sw = {"RUN_MOBILENET = False": "RUN_MOBILENET = True", "RUN_YOLO = False": "RUN_YOLO = True",
              "RUN_BYTETRACK = False": "RUN_BYTETRACK = True", "EPOCHS = 5 #": "EPOCHS = 3 #",
              "MAX_TRACK_FRAMES = 60 #": "MAX_TRACK_FRAMES = 30 #", "n_jobs=-1": "n_jobs=1"}
    nb = build(out, switches=sw)
    n_md = sum(c.cell_type == "markdown" for c in nb.cells)
    print(f"written {out}: {len(nb.cells)} cells ({n_md} text, {len(nb.cells) - n_md} code), {os.path.getsize(out) / 1e3:.0f} kB")
