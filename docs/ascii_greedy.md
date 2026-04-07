# ASCII Greedy

Este documento describe la variante greedy de arte ASCII implementada en [`main_ascii_greedy.py`](/home/jicanta/Desktop/tps-itba/sia/genetic-algorithms/main_ascii_greedy.py).

La idea es aproximar una imagen usando una grilla de caracteres, pero sin algoritmo genético. En vez de evolucionar una población, se resuelve directamente qué carácter poner en cada celda.

## 1. Objetivo

Entrada:

- una imagen cualquiera
- una fuente monoespaciada
- un conjunto de caracteres ASCII

Salida:

- una matriz de caracteres
- su render en PNG

La salida se construye bloque por bloque: para cada celda de la grilla, se elige el carácter cuyo glifo renderizado minimiza el error contra ese bloque de la imagen.

## 2. Preprocesamiento

Antes del matching se hacen dos cosas.

### 2.1. Cache de glifos

Para cada carácter del `charset`, el programa renderiza su glifo real con PIL sobre una celda fija de tamaño `(cell_h, cell_w)`.

Eso produce un tensor:

```text
glyphs.shape = (n_chars, cell_h, cell_w)
```

Cada entrada contiene la imagen en grises del carácter tal como se vería en la salida final.

### 2.2. Imagen objetivo

La imagen de entrada:

- se convierte a escala de grises
- se redimensiona exactamente al tamaño del canvas ASCII final

Si la grilla ASCII tiene `rows x cols`, entonces la imagen final queda de tamaño:

```text
(rows * cell_h, cols * cell_w)
```

Eso permite comparar pixel a pixel sin reescalar durante el matching.

## 3. Formulación exacta

La implementación actual usa directamente matching en grises contra los glifos renderizados.

Después del resize, la imagen objetivo queda dividida implícitamente en bloques de tamaño `(cell_h, cell_w)`. Cada uno de esos bloques corresponde a una celda ASCII.

El problema que se resuelve es:

- renderizar la salida final pegando glifos independientes
- medir error total como suma de errores locales por celda
- elegir, para cada celda, el carácter que minimiza ese error local

La métrica usada es SSE, o equivalentemente MSE porque todas las celdas tienen el mismo tamaño.

## 4. Algoritmo

Para cada celda `(r, c)`:

1. se extrae el bloque objetivo correspondiente
2. se compara contra todos los glifos renderizados
3. se elige el carácter con menor error cuadrático

Formalmente, si:

- `B_rc` es el bloque objetivo de la celda `(r, c)`
- `G_k` es el glifo del carácter `k`

entonces se elige:

```text
argmin_k  SSE(B_rc, G_k)
```

donde:

```text
SSE(B, G) = sum((B - G)^2)
```

Como todas las celdas tienen el mismo tamaño, minimizar `SSE` o `MSE` por celda es equivalente.

## 5. Por qué este algoritmo es óptimo

El render ASCII final se construye pegando celdas independientes. Entonces el error total contra la imagen objetivo es:

```text
Error total = suma de errores de cada celda
```

No hay solapamiento entre glifos ni interacción entre bloques en la imagen renderizada.

Entonces:

- cada término del error depende solo del carácter elegido en esa celda
- minimizar cada celda por separado minimiza la suma total

Es exactamente el mismo principio que minimizar una función separable:

```text
f(x1, ..., xn) = g1(x1) + g2(x2) + ... + gn(xn)
```

Si cada `gk` depende solo de `xk`, la solución óptima global se obtiene minimizando cada `gk` por separado.

En este problema:

- `xk` = carácter elegido en una celda
- `gk` = error del bloque correspondiente contra el glifo de ese carácter

## 6. Cuándo deja de ser óptimo

Esto deja de valer si la función objetivo agrega términos no separables, por ejemplo:

- penalización por discontinuidad entre vecinos
- restricciones globales sobre cantidad de caracteres
- suavidad espacial
- cualquier interacción entre celdas

En esos casos ya no alcanza con resolver cada bloque de forma independiente.

## 7. Implementación eficiente

El matching no compara un bloque contra cada carácter con loops de Python anidados. En cambio, se vectoriza con NumPy.

Cada bloque y cada glifo se aplana a un vector:

```text
block_vectors.shape = (rows * cols, cell_h * cell_w)
glyph_vectors.shape = (n_chars, cell_h * cell_w)
```

Luego se calcula la matriz completa de errores usando:

```text
||b - g||^2 = ||b||^2 + ||g||^2 - 2 b·g
```

Eso permite obtener, para todas las celdas y todos los caracteres, el error cuadrático sin loops explícitos sobre píxeles.

## 8. Diferencia con un greedy por brillo promedio

Esto no es lo mismo que:

1. calcular brillo promedio del bloque
2. elegir un carácter con oscuridad parecida

Ese método es solo una aproximación rápida.

La formulación exacta compara el bitmap completo del bloque contra el bitmap completo de cada glifo. Eso importa porque dos caracteres pueden tener tinta total parecida pero geometría distinta.

Ejemplos típicos:

- `-`
- `|`
- `.`
- `+`

Pueden tener coberturas similares, pero distribuciones espaciales muy diferentes.

## 9. Limitaciones prácticas

Aunque esta solución es óptima para la formulación separable, eso no significa que siempre produzca la mejor imagen perceptual.

Sigue habiendo límites inevitables:

- la resolución ASCII puede ser demasiado baja para detalles chicos
- si el resize destruye un detalle, el algoritmo no puede recuperarlo
- el criterio minimiza error numérico pixel a pixel, no semántica visual
- puede elegir caracteres raros si eso baja el error
