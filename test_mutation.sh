#!/bin/bash
# Prueba cada método de mutación en ambas imágenes.
# Variables fijas elegidas por estabilidad:
#   selección  → tournament_det  (presión selectiva consistente, poco ruido)
#   cruce      → two_point       (intercambia segmentos, buen balance exploración/explotación)
#   supervivencia → exclusive    (reemplazo total, estándar)

IMAGES=("images/test_smiley.png" "images/argentina_flag.png")
MUTATIONS=("uniform" "gen" "multigen" "non_uniform")

N_TRIANGLES=50
POPULATION=65
GENERATIONS=100
IMG_SIZE=64
SAVE_EVERY=50

for IMAGE in "${IMAGES[@]}"; do
    IMAGE_NAME=$(basename "$IMAGE" .png)
    for MUT in "${MUTATIONS[@]}"; do
        OUTPUT="output_tests/${IMAGE_NAME}/mutation_${MUT}"
        echo "=== $MUT | $IMAGE_NAME ==="
        python3 main_triangles.py "$IMAGE" \
            --mutation       "$MUT" \
            --selection      tournament_det \
            --crossover      two_point \
            --survival       exclusive \
            --n-triangles    $N_TRIANGLES \
            --population     $POPULATION \
            --generations    $GENERATIONS \
            --img-size       $IMG_SIZE \
            --save-every     $SAVE_EVERY \
            --stop-stagnation --stagnation-gens 30 --stagnation-delta 1.0 \
            --stop-convergence --convergence-thr 50.0 \
            --output         "$OUTPUT"
        echo ""
    done
done

echo "Resultados en output_tests/"
