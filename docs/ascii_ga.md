# Ejercicio 1: Algoritmo Genético Actual para Arte ASCII

Este documento describe en profundidad cómo funciona la implementación actual del ejercicio 1 en `ascii_ga/`.

La idea general es aproximar una imagen en escala de grises usando una grilla de caracteres ASCII. Cada individuo del algoritmo genético representa una imagen ASCII completa. El fitness mide qué tan parecida es esa imagen renderizada a la imagen objetivo. A partir de esa comparación, el algoritmo selecciona mejores individuos, los cruza, los muta y repite el proceso hasta mejorar progresivamente la aproximación.

## 1. Objetivo del problema

El problema que resuelve esta implementación es:

- entrada: una imagen cualquiera
- salida: una grilla de caracteres ASCII que, al renderizarse con una fuente monoespaciada, se parezca lo máximo posible a la imagen original

En lugar de hacer una conversión directa de brillo a carácter, esta implementación usa búsqueda evolutiva. Eso significa que no decide cada celda de forma totalmente independiente, sino que optimiza la imagen completa como una estructura global.

## 2. Representación del individuo

Cada individuo es una matriz de enteros de forma `(rows, cols)`.

Cada entero representa el índice de un carácter dentro del `charset`.

Ejemplo conceptual:

```text
charset = "@%#*+=-:. "

genome =
[
  [0, 0, 1, 3, 8],
  [0, 2, 4, 7, 9],
  [1, 3, 5, 8, 9],
]
```

Eso equivale a una imagen ASCII donde cada posición de la grilla contiene un carácter del conjunto permitido.

### Qué significa un gen

En este modelo:

- un gen = una celda ASCII
- el alelo del gen = qué carácter ocupa esa celda
- el cromosoma completo = toda la imagen ASCII

Esto es una decisión razonable para el ejercicio 1 porque:

- mantiene una codificación simple
- permite cruzas y mutaciones locales
- conserva estructura espacial 2D
- hace fácil renderizar y evaluar el individuo

## 3. Preprocesamiento de fuente e imagen

Antes de correr el AG, la implementación construye dos cosas clave.

### 3.1. Cache de glifos

Para cada carácter del `charset`, se renderiza una imagen pequeña en escala de grises usando una fuente monoespaciada.

Cada glifo queda guardado como un tile de tamaño fijo `(cell_h, cell_w)`.

Esto permite que, durante la evolución:

- no haya que llamar a PIL para dibujar caracteres en cada fitness
- renderizar un individuo sea solo indexar arrays de NumPy y reacomodarlos

Además, para cada carácter se calcula su `darkness`, que mide cuánta tinta ocupa:

- `0` = casi blanco
- `1` = muy negro

Ese valor se usa tanto para inicialización como para mutación.

### 3.2. Imagen objetivo

La imagen de entrada:

- se convierte a escala de grises
- se redimensiona exactamente al tamaño en píxeles del canvas ASCII final

La resolución final depende de:

- cantidad de columnas ASCII
- tamaño real de cada carácter en la fuente
- cantidad de filas derivada del aspect ratio

Importante: esta implementación no fuerza una grilla estrictamente `N x N`. Mantiene proporción visual de la imagen según el aspecto de los caracteres. Eso mejora calidad perceptual, aunque no sigue literalmente el enunciado si se interpreta como cuadrícula obligatoriamente cuadrada.

## 4. Población inicial

La población inicial mezcla explotación temprana con diversidad.

### 4.1. Warm start greedy

La mitad de la población se genera a partir de un individuo base `greedy`.

Ese individuo se construye así:

1. Se divide la imagen objetivo en celdas ASCII.
2. Para cada celda se calcula el brillo promedio.
3. Se elige el carácter cuya oscuridad medida mejor coincide con ese brillo.

Esto da una primera aproximación bastante razonable sin evolución.

### 4.2. Diversificación por ruido

Ese individuo base no se copia directamente muchas veces. Se lo muta con niveles crecientes de ruido.

La idea es:

- arrancar cerca de una solución ya usable
- evitar que todos los individuos sean casi idénticos

### 4.3. Resto aleatorio

La otra mitad de la población se crea completamente al azar.

Esto aporta exploración global y reduce el riesgo de converger demasiado rápido a la solución greedy inicial.

## 5. Función de fitness

El fitness actual es el error cuadrático medio entre:

- la imagen renderizada del individuo
- la imagen objetivo preprocesada

Formalmente:

```text
MSE = promedio((rendered - target)^2)
```

Menor valor significa mejor individuo.

### Por qué MSE

La implementación usa MSE en vez de MAE porque penaliza más los errores grandes. Eso empuja a corregir zonas donde el render ASCII se desvía mucho de la imagen objetivo.

### Cómo se evalúa un individuo

Para calcular el fitness:

1. Se toma la matriz de genes.
2. Se reemplaza cada índice por su glifo pre-renderizado.
3. Se arma la imagen completa uniendo esos tiles.
4. Se compara pixel a pixel contra la imagen objetivo.

Esto hace que el fitness no evalúe solo brillo promedio por celda, sino la forma real del carácter renderizado.

Ese punto es importante: dos caracteres con oscuridad parecida pueden dibujar texturas distintas, y el fitness real captura esa diferencia.

## 6. Selección de padres

La implementación actual usa selección por torneo determinístico.

Procedimiento:

1. Se eligen `k` individuos al azar de la población.
2. Se comparan sus fitness.
3. Se selecciona el de menor MSE.

Esto se repite cada vez que hace falta un padre.

### Efecto del parámetro `tournament_k`

- `k` chico: más exploración, menos presión selectiva
- `k` grande: más explotación, convergencia más rápida, más riesgo de pérdida de diversidad

Esta implementación no usa ruleta, ranking ni Boltzmann en `ascii_ga`. Para ejercicio 1 eso no es un problema conceptual, pero sí significa que el motor ASCII actual es una versión simplificada respecto a lo pedido para el ejercicio 2.

## 7. Cruza

Cada pareja de padres puede cruzarse con probabilidad `crossover_prob`.

Si no se cruzan:

- los hijos son copias de los padres
- luego igualmente pasan por mutación

Si se cruzan, se elige aleatoriamente uno de dos métodos.

### 7.1. Cruza uniforme

Para cada celda de la grilla:

- con probabilidad `0.5`, el hijo toma el gen del padre 1
- si no, toma el del padre 2

Ventajas:

- mezcla fina
- combina detalles locales de ambos padres
- introduce mucha variación

Desventaja:

- puede romper patrones espaciales coherentes

### 7.2. Cruza por bloque rectangular

Se elige un subrectángulo no vacío dentro de la imagen y se intercambia entre padres.

Ventajas:

- preserva estructura espacial
- es mejor para transferir regiones completas útiles
- tiene más sentido para una representación 2D que una cruza lineal clásica

La implementación actual fue corregida para asegurar que el rectángulo intercambiado no sea vacío. Antes podía ocurrir una “cruza” que en realidad no cambiaba nada.

## 8. Mutación

Después de la cruza, cada hijo se muta.

La mutación es por celda con probabilidad `mutation`.

Para las posiciones elegidas:

- 70% de las veces se aplica una mutación local guiada por oscuridad
- 30% de las veces se reemplaza por un carácter aleatorio distinto

### 8.1. Mutación local guiada por oscuridad

Cada carácter tiene un ranking de oscuridad.

La mutación local:

- toma el carácter actual
- se mueve `±1` o `±2` posiciones en ese ranking

Esto produce pequeños ajustes:

- un poco más oscuro
- un poco más claro

Es útil porque muchas mejoras en ASCII art son graduales, no saltos totalmente aleatorios.

### 8.2. Mutación aleatoria global

Con menor probabilidad, el gen salta a cualquier otro carácter del conjunto.

Eso mantiene exploración y ayuda a escapar de óptimos locales.

### 8.3. Corrección aplicada

La implementación actual fue ajustada para que una mutación elegida realmente cambie el gen.

Antes podían pasar tres cosas:

- elegir un salto `0`
- clippear contra un borde y quedar en el mismo carácter
- elegir aleatoriamente exactamente el mismo carácter actual

Eso reducía la fuerza efectiva de mutación.

## 9. Elitismo y reemplazo generacional

En cada generación:

1. Se copian sin cambios los `elite` mejores individuos.
2. El resto de la nueva población se completa con hijos generados por selección, cruza y mutación.
3. La población anterior se reemplaza por esta nueva generación.

Esto equivale a una estrategia de reemplazo elitista con preservación explícita de los mejores.

### Qué garantiza el elitismo

- el mejor individuo conocido no se pierde por una mala mutación
- el fitness best-so-far no empeora entre generaciones

### Qué costo tiene

- si `elite` es demasiado alto, sube la presión selectiva
- eso puede reducir diversidad y acelerar convergencia prematura

## 10. Criterios de corte

El algoritmo siempre tiene un máximo de generaciones.

Además, opcionalmente puede cortar por:

### 10.1. Estancamiento

Si el mejor fitness no mejora más de `stagnation_delta` durante `stagnation_gens` generaciones consecutivas, se detiene.

Esto es un criterio de contenido.

### 10.2. Convergencia

Si el desvío estándar de fitness de la población cae por debajo de `convergence_threshold`, se detiene.

Esto es un criterio estructural aproximado:

- si todos los individuos puntúan parecido
- la población probablemente perdió diversidad

No es una medición genética directa, pero sí una buena señal de colapso de la población.

## 11. Flujo completo de una ejecución

La ejecución actual sigue este pipeline:

1. Cargar la fuente monoespaciada.
2. Medir tamaño real de celda.
3. Renderizar todos los caracteres del `charset`.
4. Calcular oscuridad de cada carácter.
5. Cargar y redimensionar la imagen objetivo.
6. Construir la población inicial mezclando individuo greedy y randoms.
7. Evaluar fitness inicial.
8. Repetir por generación:
   - preservar élite
   - seleccionar padres por torneo
   - aplicar cruza con cierta probabilidad
   - mutar hijos
   - evaluar hijos
   - formar nueva población
   - actualizar mejor individuo global
   - verificar criterios de corte
9. Guardar el mejor resultado como:
   - texto ASCII
   - imagen renderizada
   - snapshots opcionales
   - GIF opcional de evolución

## 12. Por qué este diseño funciona bien para el ejercicio 1

La implementación actual tiene varias decisiones acertadas para este problema.

### Fitness sobre imagen renderizada real

Es mucho mejor que comparar solo promedios de brillo por celda, porque considera la forma concreta de cada carácter.

### Warm start

Reduce muchísimo el tiempo hasta llegar a soluciones aceptables. Empezar completamente al azar en este problema es mucho más costoso.

### Representación 2D simple

Cada gen corresponde a una celda visible. Eso vuelve intuitivas las mutaciones y las cruzas.

### Cruza mixta

Combinar uniforme con bloque permite:

- mezclar detalles
- preservar regiones útiles

### Mutación guiada por oscuridad

Tiene sentido semántico para ASCII art. No todo cambio entre caracteres es igual de razonable visualmente.

## 13. Limitaciones del algoritmo actual

Aunque la implementación es buena para el ejercicio 1, no es perfecta.

### No impone `N x N` estrictamente

Usa `rows x cols`, donde `rows` depende del aspect ratio de la imagen y la fuente.

### Optimiza solo escala de grises

No hay color.

### Evalúa todo el individuo completo en cada fitness

Eso es correcto, pero costoso. No reutiliza deltas locales.

### La selección es una sola

Solo hay torneo determinístico en `ascii_ga`.

### No hay múltiples estrategias de supervivencia

Hay elitismo con reemplazo generacional, pero no una API configurable como en `triangles_ga`.

### La convergencia se estima por fitness, no por estructura genética

Eso es práctico, pero no mide diversidad genotípica directamente.

## 14. Complejidad y costo computacional

La parte más costosa es evaluar fitness, porque cada individuo debe renderizarse completo y compararse contra la imagen objetivo.

A grandes rasgos, el costo por generación depende de:

- tamaño de población
- cantidad de celdas ASCII
- tamaño en píxeles de cada glifo

El diseño atenúa ese costo mediante:

- cache de glifos
- render vectorizado con NumPy
- representación compacta por índices

## 15. Resumen conceptual

La implementación actual del ejercicio 1 funciona así:

- representa cada imagen ASCII como una grilla de genes
- arranca desde una mezcla de soluciones razonables y aleatorias
- mide calidad comparando la imagen renderizada contra la imagen objetivo real
- selecciona mejores candidatos por torneo
- recombina tanto a nivel fino como por regiones
- muta mayormente con pequeños cambios guiados por oscuridad
- preserva élite para no perder progreso

En términos prácticos, es un AG bien alineado con el problema de arte ASCII, con una representación simple, operadores coherentes con la estructura 2D y un fitness visualmente significativo.

## 16. Archivos clave de la implementación

- `ascii_ga/config.py`: hiperparámetros del algoritmo
- `ascii_ga/font.py`: carga de fuente y construcción de glifos
- `ascii_ga/image.py`: carga y resize de la imagen objetivo
- `ascii_ga/render.py`: render vectorizado del genoma
- `ascii_ga/fitness.py`: cálculo de MSE
- `ascii_ga/operators.py`: inicialización, selección, cruza y mutación
- `ascii_ga/ga.py`: loop evolutivo, elitismo y criterios de corte
- `ascii_ga/io.py`: exportación a texto e imagen

## 17. Qué responder si te preguntan “por qué este enfoque”

Una defensa técnica corta y sólida sería:

“Modelamos cada individuo como una grilla ASCII completa porque el fenotipo final también es una grilla. Evaluamos fitness sobre la imagen renderizada real, no solo sobre brillo local, para capturar la forma de los caracteres. La población inicial combina un warm start greedy con individuos aleatorios para balancear explotación y exploración. Usamos torneo determinístico, elitismo, cruza uniforme y por bloque, y mutación guiada por oscuridad porque son operadores coherentes con una representación bidimensional y con el tipo de ajustes visuales que requiere el problema.”
