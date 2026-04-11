#!/bin/bash
# Prueba cada método de cruce en ambas imágenes.
# Variables fijas elegidas por estabilidad:
#   selección  → tournament_det  (presión selectiva consistente, poco ruido)
#   mutación   → uniform         (baseline: per-gen gaussiana, no interfiere con el cruce)
#   supervivencia → exclusive    (reemplazo total, estándar)

IMAGES=("images/test_smiley.png" "images/argentina_flag.png")
CROSSOVERS=("uniform" "one_point" "two_point" "annular")

N_TRIANGLES=50
POPULATION=65
GENERATIONS=100
IMG_SIZE=64
SAVE_EVERY=50

for IMAGE in "${IMAGES[@]}"; do
    IMAGE_NAME=$(basename "$IMAGE" .png)
    for CX in "${CROSSOVERS[@]}"; do
        OUTPUT="output_tests/${IMAGE_NAME}/crossover_${CX}"
        echo "=== $CX | $IMAGE_NAME ==="
        python3 triangles_ga/main.py "$IMAGE" \
            --crossover      "$CX" \
            --selection      tournament_det \
            --mutation       uniform \
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
