mkdir results

for i in $(seq 1 10); do
    IIDS=(0 1)
    for IID in ${IIDS[@]}; do
        # Performance
        # CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/SGD.py --epochs=40 --lr=9.0 --verbose=0
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FedAvg.py --iid=${IID} --epochs=200 --byzantines=0 --frac=0.1 --verbose=0
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FedAsync.py --iid=${IID} --epochs=200 --byzantines=0 --frac=0.1 --stale=4 --alpha=0.6 --verbose=0
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.0 --verbose=0  # same-2
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --drift=0 --interpol=slerp --adaptive=constant


        # Stale
        # LERP vs SLERP in stale env.
        ## LERP: same as BRAIN
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FedAsync.py --iid=${IID} --epochs=200 --byzantines=0 --frac=0.1 --stale=16 --alpha=0.6 --verbose=0
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=16 --diff=0.55 --window=4 --threshold=0.0 --verbose=0  # same-1
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=16 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --drift=0 --interpol=lerp --adaptive=constant  # same-1
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=16 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --drift=0 --interpol=slerp --adaptive=constant


        # Byzantine
        ## Nullifiers
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FedAvg.py --iid=${IID} --epochs=200 --byzantines=10 --frac=0.1 --verbose=0
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FedAsync.py --iid=${IID} --epochs=200 --byzantines=10 --frac=0.1 --stale=4 --alpha=0.6 --verbose=0
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=10 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=10 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=0 --interpol=slerp --adaptive=constant
        ## Distruptors (with 5 Nullifiers)
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=5 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=0 --interpol=slerp --adaptive=constant
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=5 --score_byzantines=5 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=0 --interpol=slerp --adaptive=constant
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=5 --score_byzantines=10 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=0 --interpol=slerp --adaptive=constant
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=5 --score_byzantines=11 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=0 --interpol=slerp --adaptive=constant
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=5 --score_byzantines=15 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=0 --interpol=slerp --adaptive=constant


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

        # Stale & Drift
        # LERP vs SLERP in stale and drifted env.
        ## LERP: same as BRAIN
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=16 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --drift=11 --interpol=lerp --adaptive=constant
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=16 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --drift=11 --interpol=slerp --adaptive=constant


        # \alpha
        # CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --drift=0 --interpol=slerp --adaptive=constant
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --drift=0 --interpol=slerp --adaptive=poly --adaptive_a=0.5
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --drift=0 --interpol=slerp --adaptive=hinge --adaptive_a=10.0 --adaptive_b=2.0 --adaptive_c=4.0
    done
done
