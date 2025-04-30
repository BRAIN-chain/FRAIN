mkdir results

for i in $(seq 1 10); do
    IIDS=(0 1)
    for IID in ${IIDS[@]}; do
        # Performance
        # CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/SGD.py --epochs=40 --lr=9.0 --verbose=0
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FedAvg.py --iid=${IID} --epochs=200 --byzantines=0 --frac=0.1 --verbose=0
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FedAsync.py --iid=${IID} --epochs=200 --byzantines=0 --frac=0.1 --stale=4 --alpha=0.6 --verbose=0
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.0 --verbose=0
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python src/FRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --drift=0 --interpol=slerp --adaptive=constant
    done
done
