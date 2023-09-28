import glob
from subprocess import check_output

for f in glob.glob("./**/*.ipynb", recursive=True):
    print(f"f={f}")
    cmd = ["jupytext", "--to", "script", f"{f}"]
    print(f"cmd={cmd}")
    check_output(cmd)
