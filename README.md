# TP2 - Algoritmos Genéticos

**Sistemas de Inteligencia Artificial — ITBA**

Implementación de un Algoritmo Genético para aproximar imágenes como arte ASCII. El algoritmo evoluciona una grilla completa de caracteres minimizando la diferencia visual entre la imagen renderizada y la imagen objetivo.

---

## Dependencias

```bash
pip install pillow numpy
# Opcional, para exportar GIF:
pip install imageio
```

---

## Uso

```bash
python3 main.py <imagen> [opciones]
```

### Parámetros

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--cols` | `80` | Columnas ASCII |
| `--population` | `80` | Tamaño de la población |
| `--generations` | `500` | Generaciones |
| `--mutation` | `0.02` | Tasa de mutación por celda |
| `--font` | *(auto)* | Ruta a fuente TTF monoespaciada |
| `--font-size` | `12` | Tamaño de fuente en puntos |
| `--save-every` | `50` | Guardar snapshot cada N generaciones |
| `--output` | `output/` | Directorio de salida |
| `--elite` | `5` | Individuos élite por generación |
| `--tournament-k` | `5` | Tamaño del torneo de selección |
| `--charset` | `@%#*+=-:. ` | Set de caracteres (oscuro→claro) |
| `--char-aspect` | *(auto)* | Relación cell_w/cell_h para corregir proporción |
| `--gif` | *(flag)* | Exportar `evolution.gif` |
| `--seed` | `42` | Semilla aleatoria |

### Ejemplos

```bash
# Ejecución básica
python3 main.py imagen.jpg

# Mayor resolución y más generaciones
python3 main.py imagen.jpg --cols 120 --generations 2000 --population 100

# Exportar evolución como GIF
python3 main.py imagen.jpg --save-every 25 --gif

# Fuente personalizada
python3 main.py imagen.jpg --font /usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf --font-size 10
```

---

## Cómo funciona

A diferencia de un conversor ASCII clásico (que mapea brillo local → caracter), este algoritmo hace búsqueda global:

1. **Representación:** cada individuo es una grilla de índices al charset.
2. **Renderizado:** la grilla se convierte en imagen pegando los glifos reales de la fuente.
3. **Fitness:** MSE entre la imagen renderizada y la imagen objetivo (menor = mejor).
4. **Evolución:** selección por torneo, cruce uniforme o por bloque rectangular, mutación con sesgo al vecino en orden de oscuridad.
5. **Warm start:** mitad de la población inicial viene de un mapeo greedy de brillo; la otra mitad es aleatoria.

### Operadores genéticos

- **Selección:** torneo determinístico de tamaño `k`
- **Cruce:** uniforme por celda (mezcla fina) o por bloque rectangular (preserva coherencia espacial), alternados al azar
- **Mutación:** 70% paso ±1/2 en orden de oscuridad medida (búsqueda local), 30% caracter aleatorio (exploración)
- **Elitismo:** los top-N pasan sin modificar a la siguiente generación

---

## Output

```
output/
├── best.txt            # Arte ASCII en texto plano
├── best.png            # Imagen renderizada del mejor individuo
├── evolution.gif       # Animación de la evolución (con --gif)
└── snapshots/
    ├── gen_00050.txt
    ├── gen_00050.png
    └── ...
```

---

## Estructura

```
genetic-algorithms/
├── main.py              # CLI: parseo de argumentos y loop de evolución
├── ascii_ga/
│   ├── config.py        # Config dataclass con todos los hiperparámetros
│   ├── font.py          # Carga de fuente y construcción del glyph cache
│   ├── image.py         # Carga y preprocesado de la imagen objetivo
│   ├── render.py        # render_genome: genome → imagen numpy (vectorizado)
│   ├── fitness.py       # compute_fitness: MSE imagen renderizada vs target
│   ├── operators.py     # greedy_genome, selección, cruce, mutación
│   ├── ga.py            # Clase ASCIIArtGA: población, initialize(), step()
│   └── io.py            # genome_to_text, save_result
└── images/              # Imágenes de entrada
```
