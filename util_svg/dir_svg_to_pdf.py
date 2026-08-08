import subprocess
from pathlib import Path
import constants as cons

def svg_to_pdf(svg_path: Path, pdf_path: Path):
    subprocess.run([
        "inkscape",
        str(svg_path),
        "--export-type=pdf",
        f"--export-filename={pdf_path}"
    ], check=True)

def convert_directory(input_dir: str, output_dir: str):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for svg_file in input_dir.rglob("*.svg"):
        relative = svg_file.relative_to(input_dir)
        pdf_file = (output_dir / relative).with_suffix(".pdf")
        pdf_file.parent.mkdir(parents=True, exist_ok=True)

        svg_to_pdf(svg_file, pdf_file)
        print(f"OK: {svg_file} → {pdf_file}")

# Uso
DIR = f"{cons.IMAGES_DIR}/evolution/"
convert_directory(DIR, DIR)