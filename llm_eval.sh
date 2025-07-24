mkdir results

DEVICE=0

# TODO: 1 10
for i in $(seq 1 5); do

    # TODO
    # IIDS=(0 1)
    IIDS=(0)  # only non-IID

    for IID in ${IIDS[@]}; do
        # Performance
        # CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/SGD.py --epochs=25 --verbose=0  #--epochs=50
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FedAvg.py --iid=${IID} --frac=0.1 --verbose=0  #--epochs=50
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FedAsync.py --iid=${IID} --frac=0.1 --stale=4 --alpha=0.6 --verbose=0  #--epochs=50
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/BRAIN.py --iid=${IID} --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.0 --verbose=0  #--epochs=50
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FRAIN.py --iid=${IID} --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --drift=0 --interpol=slerp --adaptive=constant  #--epochs=50

        # Byzantine - Randomizer
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FedAvg.py --iid=${IID} --byzantines=10 --frac=0.1 --verbose=0  #--epochs=50
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FedAsync.py --iid=${IID} --byzantines=10 --frac=0.1 --stale=4 --alpha=0.6 --verbose=0  #--epochs=50
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/BRAIN.py --iid=${IID} --byzantines=10 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0  #--epochs=50
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FRAIN.py --iid=${IID} --byzantines=10 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=0 --interpol=slerp --adaptive=constant  #--epochs=50
    done
done
