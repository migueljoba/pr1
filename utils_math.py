#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any
import csv
import math
import glob
import os


# =========================
# 1) Capa de EXTRACCIÓN
# =========================

def load_summary_csv(path: str | Path) -> Tuple[List[float], List[int]]:
    """
    Carga un CSV con cabeceras 'value,count' y devuelve (values, counts)
    ordenados por value ascendente.

    - value: numérico (int/float)
    - count: entero >= 0
    """
    path = Path(path)
    values: List[float] = []
    counts: List[int] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        expected = {"value", "count"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError(f"Estructura inválida en {path.name}. "
                             f"Se esperan cabeceras exactamente {expected} en cualquier orden.")

        rows = []
        for row in reader:
            v = float(row["value"])
            c = int(row["count"])
            if c < 0:
                raise ValueError(f"Count negativo en {path.name}: value={v}, count={c}")
            if c > 0:
                rows.append((v, c))

        # Ordenar por value (asc)
        rows.sort(key=lambda t: t[0])
        for v, c in rows:
            values.append(v)
            counts.append(c)

    if not values:
        raise ValueError(f"Archivo vacío o sin conteos positivos: {path}")

    return values, counts


# =========================
# 2) Capa de FÓRMULAS
# =========================

def _total(counts: Iterable[int]) -> int:
    return sum(counts)


def mean_weighted(values: Iterable[float], counts: Iterable[int]) -> float:
    """
    Media ponderada por frecuencias (counts).
    """
    values = list(values)
    counts = list(counts)
    N = _total(counts)
    if N == 0:
        raise ValueError("Total de observaciones N=0")
    return sum(v * c for v, c in zip(values, counts)) / N


def variance_weighted(values: Iterable[float], counts: Iterable[int], sample: bool = False) -> float:
    """
    Varianza ponderada por frecuencias.

    - sample=False: varianza poblacional -> sum(c*(x-mu)^2) / N
    - sample=True : varianza muestral   -> sum(c*(x-mu)^2) / (N-1)  (si N>1)
      (Válido cuando counts son frecuencias enteras)
    """
    values = list(values)
    counts = list(counts)
    N = _total(counts)
    if N <= 0:
        raise ValueError("Total de observaciones N=0")
    mu = mean_weighted(values, counts)
    s2_num = sum(c * (v - mu) ** 2 for v, c in zip(values, counts))
    if sample:
        if N < 2:
            raise ValueError("N<2 para varianza muestral")
        return s2_num / (N - 1)
    return s2_num / N


def std_weighted(values: Iterable[float], counts: Iterable[int], sample: bool = False) -> float:
    """Desviación estándar ponderada (raíz de la varianza ponderada)."""
    return math.sqrt(variance_weighted(values, counts, sample=sample))


def min_weighted(values: Iterable[float], counts: Iterable[int]) -> float:
    """Mínimo valor con count>0."""
    return min(v for v, c in zip(values, counts) if c > 0)


def max_weighted(values: Iterable[float], counts: Iterable[int]) -> float:
    """Máximo valor con count>0."""
    return max(v for v, c in zip(values, counts) if c > 0)


def modes_weighted(values: Iterable[float], counts: Iterable[int]) -> List[float]:
    """Lista de modas (puede haber más de una)."""
    values = list(values)
    counts = list(counts)
    if not values:
        return []
    m = max(counts)
    return [v for v, c in zip(values, counts) if c == m]


def _weighted_rank(values: List[float], counts: List[int], q: float) -> float:
    """
    Percentil ponderado (q de 0..100). Devuelve el valor del percentil usando
    la convención de 'observaciones expandidas' y selección por posición:
      - Posición 1-indexed: p = q/100 * (N - 1) + 1
      - Si p es entero -> el valor en esa posición
      - Si no, interpolación lineal entre los dos vecinos
    """
    if not (0.0 <= q <= 100.0):
        raise ValueError("q debe estar en [0, 100]")

    N = _total(counts)
    if N == 0:
        raise ValueError("Total de observaciones N=0")

    # Posición 1-indexed tipo Hyndman & Fan (P7) adaptada a datos discretos
    p = (q / 100.0) * (N - 1) + 1
    # Recorremos acumulados para ubicar vecinos
    cum = 0
    prev_val = values[0]
    prev_cum = 0

    for v, c in zip(values, counts):
        next_cum = cum + c
        if p <= next_cum:
            # Estamos dentro del bloque [cum+1 .. next_cum]
            if p.is_integer():
                return v
            # Interpolación entre prev y actual (si hay separación)
            if cum == prev_cum:
                # No hay vecino previo distinto, devolver v
                return v
            # Fracción entre los dos puntos prev_val y v
            # Situamos la posición fraccional dentro del segmento [cum, next_cum]
            # y usamos interpolación entre prev_val y v
            left_pos = cum
            right_pos = next_cum
            frac = (p - left_pos) / (right_pos - left_pos)
            return prev_val + (v - prev_val) * frac
        prev_val = v
        prev_cum = cum
        cum = next_cum

    # Si por un borde numérico no entró al loop, devolver el máximo
    return values[-1]


def percentile_weighted(values: Iterable[float], counts: Iterable[int], q: float) -> float:
    """Wrapper público para percentil ponderado (q en [0,100])."""
    v = list(values)
    c = list(counts)
    if len(v) != len(c):
        raise ValueError("values y counts deben tener la misma longitud")
    return _weighted_rank(v, c, q)


def median_weighted(values: Iterable[float], counts: Iterable[int]) -> float:
    """Mediana ponderada (percentil 50)."""
    return percentile_weighted(values, counts, 50.0)


def quantiles_weighted(values: Iterable[float], counts: Iterable[int], qs: Iterable[float]) -> Dict[float, float]:
    """Varios percentiles ponderados de una sola vez."""
    v = list(values)
    c = list(counts)
    return {q: percentile_weighted(v, c, q) for q in qs}


def harmonic_mean_weighted(values: Iterable[float], counts: Iterable[int]) -> float:
    """Media armónica ponderada por frecuencias (requiere valores > 0)."""
    values = list(values)
    counts = list(counts)
    if any(v <= 0 for v in values if v is not None):
        raise ValueError("Media armónica requiere valores > 0")
    N = _total(counts)
    denom = sum(c / v for v, c in zip(values, counts))
    if denom == 0:
        raise ValueError("Denominador cero en media armónica")
    return N / denom


def geometric_mean_weighted(values: Iterable[float], counts: Iterable[int]) -> float:
    """Media geométrica ponderada por frecuencias (requiere valores > 0)."""
    values = list(values)
    counts = list(counts)
    if any(v <= 0 for v in values):
        raise ValueError("Media geométrica requiere valores > 0")
    N = _total(counts)
    # log-espacio para estabilidad
    log_sum = sum(c * math.log(v) for v, c in zip(values, counts))
    return math.exp(log_sum / N)


# =========================
# 3) Ejemplo de uso (opcional)
# =========================

def summarize_stats(values: List[float], counts: List[int]) -> Dict[str, Any]:
    """Convenience para obtener varias métricas de una vez (útil en notebooks o pruebas rápidas)."""
    mu = mean_weighted(values, counts)
    stats = {
        "min": min_weighted(values, counts),
        "q25": percentile_weighted(values, counts, 25.0),
        "median": median_weighted(values, counts),
        "q75": percentile_weighted(values, counts, 75.0),
        "max": max_weighted(values, counts),
        "mean": mu,
        "var_pop": variance_weighted(values, counts, sample=False),
        "std_pop": std_weighted(values, counts, sample=False),
    }
    # Armónica y geométrica solo si aplican
    try:
        stats["hmean"] = harmonic_mean_weighted(values, counts)
    except Exception:
        pass
    try:
        stats["gmean"] = geometric_mean_weighted(values, counts)
    except Exception:
        pass
    return stats


if __name__ == "__main__":
    # Ejemplo mínimo: aplica al archivo de resumen 'summary-*.csv' d
    filename = "summary-dim20-prob-d0.2c0.8-radio1-pay1-factor2.0-tol67.csv"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_csv = os.path.join(script_dir, "generador_datos", "simulations", "summaries", filename)

    path = Path(path_csv)
    if path.exists():
        vals, cnts = load_summary_csv(path_csv)
        res = summarize_stats(vals, cnts)
        print("Estadísticas:", res)
    else:
        print("Coloca un archivo 'summary-ejemplo.csv' junto al script para probar rápidamente.")
