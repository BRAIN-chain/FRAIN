mkdir results

for i in $(seq 1 10); do
    IIDS=(0 1)
    for IID in ${IIDS[@]}; do
        # Byzantine
        ## Distruptors (with 5 Nullifiers)
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=5 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=0 --interpol=slerp --adaptive=constant
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=5 --score_byzantines=5 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=0 --interpol=slerp --adaptive=constant
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=5 --score_byzantines=10 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=0 --interpol=slerp --adaptive=constant
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=5 --score_byzantines=11 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=0 --interpol=slerp --adaptive=constant
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=5 --score_byzantines=15 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=0 --interpol=slerp --adaptive=constant
    done
done
