#!/usr/bin/env python3
from pathlib import Path

import pandas as pd

# --- Configuración ---
# Directorio de entrada: por defecto, el mismo donde está este script
BASE_DIR = Path(__file__).resolve().parent

# Directorio de entrada
INPUT_DIR = BASE_DIR / "simulations"

# Directorio de salida (se creará si no existe)
OUTPUT_DIR = BASE_DIR / "summaries"

# Prefijo para los archivos generados (evita re-procesarlos)
OUTPUT_PREFIX = "summary-"


def summarize_info_generations(csv_path: Path) -> pd.DataFrame:
    """Devuelve DataFrame con columnas ['value','count'] para info_generations."""
    df = pd.read_csv(csv_path)
    if "info_generations" not in df.columns:
        raise ValueError(f"Falta columna 'info_generations' en: {csv_path.name}")

    # df = df[df["status_undetermined"] == False]

    # Sobrescribir info_generations con 49 donde status_undetermined == True
    df.loc[df["status_undetermined"] == True, "info_generations"] = 49

    # Asegurar numérico y eliminar NaN
    s = pd.to_numeric(df["info_generations"], errors="coerce").dropna().astype(int)

    counts = s.value_counts().sort_index()
    out = counts.rename_axis("value").reset_index(name="count")

    # Tipos enteros “limpios”
    out["value"] = out["value"].astype(int)
    out["count"] = out["count"].astype(int)
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(
        p for p in INPUT_DIR.rglob("*.csv")
        if p.is_file()
        and not p.name.startswith(OUTPUT_PREFIX)  # evita re-procesar resúmenes
    )

    if not csv_files:
        raise FileNotFoundError(f"No se encontraron CSV en {BASE_DIR}")

    for src in csv_files:
        out_name = f"{OUTPUT_PREFIX}{src.stem}.csv"
        out_path = OUTPUT_DIR / out_name

        summary_df = summarize_info_generations(src)
        # Guardar sin índice, saltos de línea consistentes
        summary_df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
