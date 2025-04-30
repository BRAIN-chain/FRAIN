mkdir results

for i in $(seq 1 10); do
    IIDS=(0 1)
    for IID in ${IIDS[@]}; do
        # Stale & Drift
        # LERP vs SLERP in stale and drifted env.
        ## LERP: same as BRAIN
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=16 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --drift=11 --interpol=lerp --adaptive=constant
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=16 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --drift=11 --interpol=slerp --adaptive=constant
    done
done
