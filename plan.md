# TP2 - Algoritmos Genéticos

## Ejercicio 1: ASCII Art (implementado como ejercicio de aprendizaje)

### Problema
Tomar una imagen cuadrada y representarla como un mapa de NxN caracteres ASCII.

### Diseño del individuo
- **Genoma:** array plano de `N*N` enteros, cada uno un índice en el charset ASCII
- **Charset:** `" .:-=+*#%@"` (ordenado de más claro a más oscuro)
- Ejemplo N=4: `[0, 5, 3, 9, 2, 0, 8, 1, ...]` → grid 4x4 de chars

### Función de aptitud
1. Renderizar el grid de chars a una imagen PIL (monocromática)
2. Redimensionar la imagen target al mismo tamaño
3. Fitness = `1 / (1 + MSE)` entre ambas imágenes (en escala de grises)

### Operadores
- **Mutación:** cambiar `k` genes aleatorios a chars aleatorios
- **Cruza:** un punto sobre el array plano de genes
- **Selección:** ruleta (proporcional al fitness)

---

## Ejercicio 2: Aproximación con Triángulos

### Diseño del individuo
- **Genoma:** array plano de `N * 10` floats en [0, 1]
- Por triángulo: `[x1, y1, x2, y2, x3, y3, R, G, B, A]` (normalizado)

### Función de aptitud
- Renderizar triángulos sobre canvas blanco (Pillow RGBA)
- Fitness = `1 / (1 + MSE)` entre rendered y target (RGB)

### Métodos de selección (todos implementados)
| Método | Descripción |
|--------|-------------|
| Elite | Toma los top-k directamente |
| Ruleta | Proporcional al fitness |
| Universal | SUS - punteros equidistantes |
| Boltzmann | Softmax con temperatura decreciente |
| Torneo Det. | Toma el mejor de m elegidos al azar |
| Torneo Prob. | Entre 2, mejor con prob p |
| Ranking | Probabilidad lineal por rango |

### Métodos de cruza (todos implementados)
- Un punto, Dos puntos, Uniforme, Anular

### Métodos de mutación (todos implementados)
- Gen (1 gen), MultiGen (prob por gen), Uniforme, No Uniforme (enfriamiento)

### Estrategias de supervivencia
- **Aditiva:** top-N de (padres + hijos)
- **Exclusiva:** hijos reemplazan completamente a padres

### Criterios de terminación
- Máximo de generaciones
- Estancamiento de contenido (fitness sin mejora por k generaciones)
- Estancamiento de estructura (genoma sin cambio por k generaciones)

---

## Estructura de archivos

```
genetic-algorithms/
├── plan.md
├── config/
│   └── default.json
├── src/
│   ├── ex1/                  # Ejercicio 1: ASCII Art
│   │   ├── renderer.py       # Renderiza grid de chars a imagen
│   │   ├── individual.py     # Genoma + inicialización random
│   │   ├── fitness.py        # MSE entre rendered y target
│   │   └── ga.py             # GA loop simple
│   ├── genetic/              # Ejercicio 2: Motor GA completo
│   │   ├── individual.py
│   │   ├── selection.py
│   │   ├── crossover.py
│   │   ├── mutation.py
│   │   ├── survival.py
│   │   ├── termination.py
│   │   └── engine.py
│   ├── rendering/
│   │   └── renderer.py       # Renderiza triángulos con Pillow
│   └── analysis/
│       └── plots.py          # Matplotlib/Plotly métricas
├── images/
├── output/
├── ex1_main.py
├── main.py
└── README.md
```

---

## Cómo correr

```bash
# Ejercicio 1
python ex1_main.py --image images/smile.png --n 32

# Ejercicio 2
python main.py --image images/flag.png --triangles 50 --config config/default.json
```
