mkdir results

for i in $(seq 1 10); do
    IIDS=(0 1)
    for IID in ${IIDS[@]}; do
        # Fast-Sync (Drift)
        # LERP vs SLERP in drifted env.
        ## LERP: same as BRAIN
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --drift=0 --interpol=lerp --adaptive=constant  # same-2 BRAIN
        # CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --drift=0 --interpol=slerp --adaptive=constant  # FRAIN
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --drift=5 --interpol=lerp --adaptive=constant  # BRAIN
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --drift=5 --interpol=slerp --adaptive=constant  # FRAIN
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --drift=11 --interpol=lerp --adaptive=constant  # BRAIN
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --drift=11 --interpol=slerp --adaptive=constant  # FRAIN
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --drift=15 --interpol=lerp --adaptive=constant  # BRAIN
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --drift=15 --interpol=slerp --adaptive=constant  # FRAIN
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --drift=21 --interpol=lerp --adaptive=constant  # BRAIN
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --drift=21 --interpol=slerp --adaptive=constant  # FRAIN
    done
done
