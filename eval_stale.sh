mkdir results

for i in $(seq 1 10); do
    IIDS=(0 1)
    for IID in ${IIDS[@]}; do
        # Stale
        # LERP vs SLERP in stale env.
        ## LERP: same as BRAIN
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FedAsync.py --iid=${IID} --epochs=200 --byzantines=0 --frac=0.1 --stale=16 --alpha=0.6 --verbose=0
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=16 --diff=0.55 --window=4 --threshold=0.0 --verbose=0  # same-1
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=16 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --drift=0 --interpol=lerp --adaptive=constant  # same-1
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=16 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --drift=0 --interpol=slerp --adaptive=constant
    done
done
