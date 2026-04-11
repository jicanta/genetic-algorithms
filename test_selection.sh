#!/bin/bash
# Prueba cada método de selección en ambas imágenes.
# Parámetros reducidos para que cada corrida tarde poco.

IMAGES=("images/test_smiley.png" "images/argentina_flag.png")
SELECTIONS=("tournament_det" "tournament_prob" "roulette" "universal" "boltzmann" "ranking")

N_TRIANGLES=50
POPULATION=65
GENERATIONS=100
IMG_SIZE=64
SAVE_EVERY=50

for IMAGE in "${IMAGES[@]}"; do
    IMAGE_NAME=$(basename "$IMAGE" .png)
    for SEL in "${SELECTIONS[@]}"; do
        OUTPUT="output_tests/${IMAGE_NAME}/selection_${SEL}"
        echo "=== $SEL | $IMAGE_NAME ==="
        python3 triangles_ga/main.py "$IMAGE" \
            --selection      "$SEL" \
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
