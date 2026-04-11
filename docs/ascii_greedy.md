# ASCII Greedy

Este documento describe la variante greedy implementada en [`ascii_ga/main_greedy.py`](/home/jicanta/Desktop/tps-itba/sia/genetic-algorithms/ascii_ga/main_greedy.py).

La idea es aproximar una imagen como arte ASCII sin algoritmo genético, pero usando una heurística visual más fuerte que el matching puro por SSE.

Entry point:

- [`ascii_ga/main_greedy.py`](/home/jicanta/Desktop/tps-itba/sia/genetic-algorithms/ascii_ga/main_greedy.py)

Comandos típicos:

```bash
python ascii_ga/main_greedy.py images/photo.jpg
python ascii_ga/main_greedy.py images/photo.jpg --cols 120
python ascii_ga/main_greedy.py images/photo.jpg --edge-weight 0 --neighbor-weight 0 --dither-strength 0
```

## 1. Objetivo

Entrada:

- una imagen objetivo
- una fuente monoespaciada
- un charset permitido

Salida:

- una grilla ASCII
- su render PNG

La construcción sigue siendo greedy, pero el criterio de elección de cada celda combina tono, estructura y continuidad local.

## 2. Preprocesamiento

### 2.1. Cache de glifos

Cada carácter del `charset` se renderiza una vez en una celda fija `(cell_h, cell_w)`.

Eso produce:

```text
glyphs.shape = (n_chars, cell_h, cell_w)
```

Cada glifo se guarda como imagen en grises.

### 2.2. Imagen objetivo

La imagen de entrada:

- se convierte a escala de grises
- se redimensiona al tamaño final del canvas ASCII

Si la grilla tiene `rows x cols`, entonces la imagen final queda de tamaño:

```text
(rows * cell_h, cols * cell_w)
```

Luego se parte en bloques no superpuestos de tamaño `(cell_h, cell_w)`.

## 3. Score híbrido por celda

Para cada bloque del target y para cada glifo candidato se calcula:

```text
score = alpha * tone_error
      + beta  * edge_error
      + gamma * neighbor_inconsistency
```

donde:

- `tone_error` compara intensidades en grises pixel a pixel
- `edge_error` compara estructura local usando Sobel `gx` y `gy`
- `neighbor_inconsistency` penaliza transiciones feas respecto de los vecinos ya elegidos

### 3.1. Tone Error

Es el error cuadrático medio entre el bloque actual y el glifo candidato:

```text
tone_error = mean((block - glyph)^2)
```

Esto sigue capturando la parte más importante: densidad y forma general del carácter.

### 3.2. Edge Error

El tono por sí solo no distingue bien orientación. Por ejemplo:

- `-` sugiere borde horizontal
- `|` sugiere borde vertical
- `/` y `\` sugieren diagonales

Por eso se calcula un descriptor Sobel para cada bloque y para cada glifo:

```text
edge_feature = [gx, gy]
```

Luego se compara con MSE:

```text
edge_error = mean((edge_block - edge_glyph)^2)
```

Esto hace que el greedy prefiera caracteres cuya estructura local se parezca a la de la imagen.

### 3.3. Neighbor Inconsistency

Si cada celda se eligiera solo por tono y borde, podrían aparecer uniones raras entre caracteres contiguos.

Entonces, al elegir la celda `(r, c)`, se consideran los vecinos ya fijados:

- izquierda `(r, c-1)`
- arriba `(r-1, c)`

La penalización compara el salto entre bordes de glifos con el salto real en la imagen objetivo a través de esa frontera.

Idea:

- si en el target la frontera es suave, se prefieren glifos con transición suave
- si en el target hay una discontinuidad real, se permite una transición fuerte

## 4. Dithering por difusión de error

Después de elegir un glifo para una celda, se calcula el residual:

```text
residual = block_actual - glyph_elegido
```

Ese residual se difunde a celdas futuras usando pesos estilo Floyd-Steinberg sobre la grilla de bloques:

- derecha: `7/16`
- abajo-izquierda: `3/16`
- abajo: `5/16`
- abajo-derecha: `1/16`

Eso no cambia el glifo ya elegido, pero modifica el bloque de trabajo de las próximas celdas para compensar error acumulado. En la práctica funciona como un dithering a nivel celda.

## 5. Orden de resolución

La grilla se resuelve en orden fila por fila, de izquierda a derecha y de arriba hacia abajo.

Ese orden es importante porque:

- los vecinos izquierdo y superior ya están decididos
- la difusión de error solo se propaga a celdas futuras

## 6. Diferencia con el greedy separable exacto

Si el score fuera solo:

```text
score = tone_error
```

entonces el problema sería separable por celdas y la solución por bloque sería globalmente óptima para ese objetivo.

Pero al agregar:

- bordes
- consistencia con vecinos
- dithering

el problema deja de ser separable.

Entonces esta versión ya no es “óptima” en sentido matemático para una función aditiva simple. Lo que gana es calidad perceptual.

## 7. Complejidad

Si:

- hay `R * C` celdas
- hay `K` caracteres
- cada glifo mide `H * W`

entonces el costo dominante sigue siendo comparar cada bloque contra todos los glifos:

```text
O(R * C * K * H * W)
```

El término de vecinos agrega poco costo extra porque solo usa fronteras izquierda y superior.

## 8. Parámetros útiles

El script expone estos pesos:

- `--tone-weight`
- `--edge-weight`
- `--neighbor-weight`
- `--dither-strength`

Defaults actuales:

- `tone_weight = 1.0`
- `edge_weight = 0.20`
- `neighbor_weight = 0.10`
- `dither_strength = 0.15`

Valores más altos de `edge` y `neighbor` suelen mejorar orientación y continuidad, pero si son demasiado altos pueden empeorar el MSE global o introducir patrones artificiales.

Si se usan:

```text
edge_weight = 0
neighbor_weight = 0
dither_strength = 0
```

entonces el algoritmo se reduce al matching puro por SSE por bloque.

Ese modo exacto sirve como referencia útil para comparar cuánto aportan realmente las heurísticas perceptuales en una imagen dada.

## 9. Limitaciones

Aunque visualmente suele ser mejor que el matching puro por tono:

- sigue sin modelar semántica global
- depende bastante de la resolución ASCII
- puede necesitar ajuste de pesos según la imagen
- detalles muy chicos pueden perderse igual si el resize ya los destruyó
