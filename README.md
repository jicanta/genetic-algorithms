# TP2 - Algoritmos Genéticos

**Sistemas de Inteligencia Artificial — ITBA**

Implementación de un motor de Algoritmos Genéticos aplicado a:
- **Ejercicio 1:** Representación de imágenes como arte ASCII
- **Ejercicio 2:** Aproximación de imágenes mediante triángulos (WIP)

---

## Dependencias

```bash
pip install pillow numpy matplotlib pygame
```

---

## Ejercicio 1: ASCII Art

Toma una imagen y la representa como una grilla NxN de caracteres ASCII, evolucionando la solución mediante un Algoritmo Genético.

### Ejecución

```bash
# Con imagen propia
python3 ex1_main.py --image <ruta_imagen> --n <tamaño_grilla>

# Sin imagen (genera un smiley de prueba automáticamente)
python3 ex1_main.py
```

### Parámetros

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--image` | *(auto)* | Ruta a la imagen de entrada |
| `--n` | `24` | Tamaño de la grilla NxN |
| `--cell` | `8` | Píxeles por celda |
| `--pop` | `60` | Tamaño de la población |
| `--gens` | `200` | Número de generaciones |
| `--mut` | `0.02` | Tasa de mutación (0.0 – 1.0) |
| `--seed` | `42` | Semilla aleatoria |
| `--no-plot` | *(flag)* | Desactiva la visualización en tiempo real |

### Ejemplos

```bash
# Grilla 32x32, 300 generaciones, visualización en vivo
python3 ex1_main.py --image images/flag.png --n 32 --gens 300

# Sin visualización (más rápido, útil para experimentar)
python3 ex1_main.py --image images/smile.png --n 20 --gens 500 --no-plot

# Población grande, alta presión selectiva
python3 ex1_main.py --n 24 --pop 120 --gens 300 --mut 0.01
```

### Output

Los resultados se guardan en `output/`:
- `ex1_result.png` — imagen renderizada del mejor individuo
- `ex1_result.txt` — arte ASCII en texto plano
- `ex1_fitness.png` — gráfico de evolución del fitness

---

## Estructura del proyecto

```
genetic-algorithms/
├── ex1_main.py          # Entry point Ejercicio 1
├── main.py              # Entry point Ejercicio 2 (WIP)
├── plan.md              # Plan de implementación
├── src/
│   ├── ex1/
│   │   ├── renderer.py  # Renderiza grilla ASCII → imagen PIL
│   │   ├── individual.py # Representación del genoma
│   │   ├── fitness.py   # Función de aptitud (MSE normalizado)
│   │   └── ga.py        # Loop del AG (selección, cruza, mutación)
│   ├── genetic/         # Motor AG completo para Ejercicio 2
│   └── rendering/       # Renderizado de triángulos
├── images/              # Imágenes de entrada
└── output/              # Resultados generados
```
