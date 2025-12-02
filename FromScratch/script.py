import json
src = "../WorkingON/reference.ipynb"
dst = "transformers_from_scratch.ipynb"
cutoff_title = "Pre-Training and Transfer Learning with Hugging Face and OpenAI"

with open(src, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = []
for cell in nb.get("cells", []):
    if (
        cell.get("cell_type") == "markdown"
        and any(cutoff_title in line for line in cell.get("source", []))
    ):
        break
    cells.append(cell)

nb["cells"] = cells

with open(dst, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Wrote", dst, "with", len(cells), "cells")

