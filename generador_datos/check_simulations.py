from pathlib import Path
import utils_file as uf
import pandas as pd


def verificar_seed(path: Path, seed_min: int = 0, seed_max: int = 1999) -> list[str]:
    """
    Verifica que la columna 'info_seed' de un CSV contenga todos los valores únicos
    en el rango [seed_min, seed_max].
    Retorna lista de errores (vacía si está todo bien).
    """
    errores = []
    df = pd.read_csv(path)

    if "info_seed" not in df.columns:
        return [f"{path.name}: no existe columna 'info_seed'"]

    seeds = df["info_seed"].to_list()

    # Verificar duplicados
    if len(seeds) != len(set(seeds)):
        duplicados = [s for s in seeds if seeds.count(s) > 1]
        errores.append(f"{path.name}: columna 'info_seed' tiene duplicados: {set(duplicados)}")

    # Verificar rango completo
    esperado = set(range(seed_min, seed_max + 1))
    presente = set(seeds)
    faltantes = esperado - presente
    extra = presente - esperado

    if faltantes:
        errores.append(f"{path.name}: faltan valores en 'info_seed': {sorted(faltantes)}")
    if extra:
        errores.append(f"{path.name}: hay valores fuera de rango en 'info_seed': {sorted(extra)}")

    return errores


if __name__ == '__main__':
    BASE_DIR = Path(__file__).resolve().parent
    SOURCE_DIR = BASE_DIR / 'simulations'

    files = uf.get_all_csv_files(SOURCE_DIR)

    for f in files:
        errores = verificar_seed(f)

        if errores:
            for e in errores:
                print("ERROR:", e)
        else:
            print(f"{f.name}: OK ✅")

