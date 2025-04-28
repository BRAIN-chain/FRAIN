mkdir results

for i in $(seq 1 10); do
    IIDS=(0 1)
    for IID in ${IIDS[@]}; do
        # Performance
        python src/SGD.py --epochs=200 --lr=9.0 --verbose=0
        python src/FedAvg.py --iid=${IID} --epochs=200 --byzantines=0 --frac=0.1 --verbose=0
        python src/FedAsync.py --iid=${IID} --epochs=200 --byzantines=0 --frac=0.1 --stale=4 --alpha=0.6 --verbose=0
        python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.0 --verbose=0

        # Byzantine
        ## FedAvg
        python src/FedAvg.py --iid=${IID} --epochs=200 --byzantines=5 --frac=0.1 --verbose=0
        python src/FedAvg.py --iid=${IID} --epochs=200 --byzantines=10 --frac=0.1 --verbose=0
        python src/FedAvg.py --iid=${IID} --epochs=200 --byzantines=11 --frac=0.1 --verbose=0
        python src/FedAvg.py --iid=${IID} --epochs=200 --byzantines=15 --frac=0.1 --verbose=0
        ## FedAsync
        python src/FedAsync.py --iid=${IID} --epochs=200 --byzantines=5 --frac=0.1 --stale=4 --alpha=0.6 --verbose=0
        python src/FedAsync.py --iid=${IID} --epochs=200 --byzantines=10 --frac=0.1 --stale=4 --alpha=0.6 --verbose=0
        python src/FedAsync.py --iid=${IID} --epochs=200 --byzantines=11 --frac=0.1 --stale=4 --alpha=0.6 --verbose=0
        python src/FedAsync.py --iid=${IID} --epochs=200 --byzantines=15 --frac=0.1 --stale=4 --alpha=0.6 --verbose=0
        ## BRAIN
        python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=5 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0
        python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=10 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0
        python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=11 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0
        python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=15 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0

        # Threshold
        python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=10 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.0 --verbose=0
        # python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=10 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0
        python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=10 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.3 --verbose=0
        python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=10 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.4 --verbose=0
        python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=10 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.5 --verbose=0
        python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=10 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.6 --verbose=0

        # Score Byzantine
        # ## Byzantine 0
        # python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=5 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.0 --verbose=0
        # python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=10 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.0 --verbose=0
        # python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=11 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.0 --verbose=0
        # python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=15 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.0 --verbose=0
        ## Byzantine 5
        python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=5 --score_byzantines=5 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.2 --verbose=0
        python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=5 --score_byzantines=10 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.2 --verbose=0
        python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=5 --score_byzantines=11 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.2 --verbose=0
        python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=5 --score_byzantines=15 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.2 --verbose=0

        # Staleness
        ## FedAsync
        python src/FedAsync.py --iid=${IID} --epochs=200 --byzantines=0 --frac=0.1 --stale=8 --alpha=0.6 --verbose=0
        python src/FedAsync.py --iid=${IID} --epochs=200 --byzantines=0 --frac=0.1 --stale=16 --alpha=0.6 --verbose=0
        python src/FedAsync.py --iid=${IID} --epochs=200 --byzantines=0 --frac=0.1 --stale=32 --alpha=0.6 --verbose=0
        # python src/FedAsync.py --iid=${IID} --epochs=200 --byzantines=0 --frac=0.1 --stale=64 --alpha=0.6 --verbose=0
        ## BRAIN
        python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=8 --diff=0.55 --window=4 --threshold=0.0 --verbose=0
        python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=16 --diff=0.55 --window=4 --threshold=0.0 --verbose=0
        python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=32 --diff=0.55 --window=4 --threshold=0.0 --verbose=0
        # python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=64 --diff=0.55 --window=4 --threshold=0.0 --verbose=0

        # Quorum
        python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=5 --score_byzantines=10 --frac=0.1 --stale=4 --diff=0.25 --window=4 --threshold=0.2 --verbose=0
        python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=5 --score_byzantines=10 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0
        python src/BRAIN.py --iid=${IID} --epochs=200 --byzantines=5 --score_byzantines=10 --frac=0.1 --stale=4 --diff=0.75 --window=4 --threshold=0.2 --verbose=0
    done
done
