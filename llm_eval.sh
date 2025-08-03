#!/usr/bin/env bash


mkdir results


DEVICE=${1:-0}
echo "Using CUDA device: ${DEVICE}"


export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1


for i in $(seq 1 2); do

    # TODO
    # IIDS=(0 1)
    IIDS=(0)  # only non-IID

    for IID in ${IIDS[@]}; do


        # Performance
        # CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/SGD.py --epochs=25 --verbose=0
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FedAvg.py   --iid=${IID} --byzantines=0  --frac=0.1 --verbose=0
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FedAsync.py --iid=${IID} --byzantines=0  --frac=0.1                       --stale=4 --alpha=0.6 --verbose=0
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/BRAIN.py    --iid=${IID} --byzantines=0  --score_byzantines=0  --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FRAIN.py    --iid=${IID} --byzantines=0  --score_byzantines=0  --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=0 --interpol=slerp --adaptive=hinge --adaptive_a=10.0 --adaptive_b=4.0 --adaptive_c=16.0


        # Byzantine
        ## Randomizer
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FedAvg.py   --iid=${IID} --byzantines=10 --frac=0.1 --verbose=0
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FedAsync.py --iid=${IID} --byzantines=10 --frac=0.1                       --stale=4 --alpha=0.6 --verbose=0
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/BRAIN.py    --iid=${IID} --byzantines=10 --score_byzantines=0  --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FRAIN.py    --iid=${IID} --byzantines=10 --score_byzantines=0  --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=0 --interpol=slerp --adaptive=hinge --adaptive_a=10.0 --adaptive_b=4.0 --adaptive_c=16.0
        ## Distruptors (with 5 Randomizer)
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/BRAIN.py    --iid=${IID} --byzantines=5  --score_byzantines=5  --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FRAIN.py    --iid=${IID} --byzantines=5  --score_byzantines=5  --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=0 --interpol=slerp --adaptive=hinge --adaptive_a=10.0 --adaptive_b=4.0 --adaptive_c=16.0


        # LERP vs SLERP
        # CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FRAIN.py    --iid=${IID} --byzantines=0  --score_byzantines=0  --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=11 --interpol=lerp --adaptive=constant  # BRAIN
        # CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FRAIN.py    --iid=${IID} --byzantines=0  --score_byzantines=0  --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=11 --interpol=slerp --adaptive=constant
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FRAIN.py    --iid=${IID} --byzantines=0  --score_byzantines=0  --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=11 --interpol=lerp  --adaptive=hinge --adaptive_a=10.0 --adaptive_b=4.0 --adaptive_c=16.0
        # CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FRAIN.py    --iid=${IID} --byzantines=0  --score_byzantines=0  --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=11 --interpol=slerp --adaptive=hinge --adaptive_a=10.0 --adaptive_b=4.0 --adaptive_c=16.0


        # Staleness Panelty Functions
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FRAIN.py    --iid=${IID} --byzantines=0  --score_byzantines=0  --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=11 --interpol=slerp --adaptive=constant
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FRAIN.py    --iid=${IID} --byzantines=0  --score_byzantines=0  --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=11 --interpol=slerp --adaptive=poly  --adaptive_a=0.5
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FRAIN.py    --iid=${IID} --byzantines=0  --score_byzantines=0  --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=11 --interpol=slerp --adaptive=hinge --adaptive_a=10.0 --adaptive_b=2.0 --adaptive_c=16.0
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FRAIN.py    --iid=${IID} --byzantines=0  --score_byzantines=0  --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=11 --interpol=slerp --adaptive=hinge --adaptive_a=10.0 --adaptive_b=4.0 --adaptive_c=16.0
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FRAIN.py    --iid=${IID} --byzantines=0  --score_byzantines=0  --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=11 --interpol=slerp --adaptive=hinge --adaptive_a=10.0 --adaptive_b=8.0 --adaptive_c=16.0


        # Fast-Sync (Drift)
        # CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FRAIN.py    --iid=${IID} --byzantines=0  --score_byzantines=0  --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=0 --interpol=lerp  --adaptive=constant  # BRAIN
        # CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FRAIN.py    --iid=${IID} --byzantines=0  --score_byzantines=0  --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=0 --interpol=slerp --adaptive=hinge --adaptive_a=10.0 --adaptive_b=4.0 --adaptive_c=16.0  # FRAIN
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FRAIN.py    --iid=${IID} --byzantines=0  --score_byzantines=0  --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=5  --interpol=lerp  --adaptive=constant  # BRAIN
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FRAIN.py    --iid=${IID} --byzantines=0  --score_byzantines=0  --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=5  --interpol=slerp --adaptive=hinge --adaptive_a=10.0 --adaptive_b=4.0 --adaptive_c=16.0  # FRAIN
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FRAIN.py    --iid=${IID} --byzantines=0  --score_byzantines=0  --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=11 --interpol=lerp  --adaptive=constant  # BRAIN
        # CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FRAIN.py    --iid=${IID} --byzantines=0  --score_byzantines=0  --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=11 --interpol=slerp --adaptive=hinge --adaptive_a=10.0 --adaptive_b=4.0 --adaptive_c=16.0  # FRAIN
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FRAIN.py    --iid=${IID} --byzantines=0  --score_byzantines=0  --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=15 --interpol=lerp  --adaptive=constant  # BRAIN
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FRAIN.py    --iid=${IID} --byzantines=0  --score_byzantines=0  --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=15 --interpol=slerp --adaptive=hinge --adaptive_a=10.0 --adaptive_b=4.0 --adaptive_c=16.0  # FRAIN
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FRAIN.py    --iid=${IID} --byzantines=0  --score_byzantines=0  --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=21 --interpol=lerp  --adaptive=constant  # BRAIN
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FRAIN.py    --iid=${IID} --byzantines=0  --score_byzantines=0  --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=21 --interpol=slerp --adaptive=hinge --adaptive_a=10.0 --adaptive_b=4.0 --adaptive_c=16.0  # FRAIN
        ## FRAIN, with 5 Distruptors (with 5 Nullifiers)
        # CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FRAIN.py    --iid=${IID} --byzantines=5  --score_byzantines=5  --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=0 --interpol=slerp --adaptive=hinge --adaptive_a=10.0 --adaptive_b=4.0 --adaptive_c=16.0
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FRAIN.py    --iid=${IID} --byzantines=5  --score_byzantines=5  --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=5 --interpol=slerp --adaptive=hinge --adaptive_a=10.0 --adaptive_b=4.0 --adaptive_c=16.0
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FRAIN.py    --iid=${IID} --byzantines=5  --score_byzantines=5  --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=11 --interpol=slerp --adaptive=hinge --adaptive_a=10.0 --adaptive_b=4.0 --adaptive_c=16.0
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FRAIN.py    --iid=${IID} --byzantines=5  --score_byzantines=5  --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=15 --interpol=slerp --adaptive=hinge --adaptive_a=10.0 --adaptive_b=4.0 --adaptive_c=16.0
        CUDA_VISIBLE_DEVICES=${DEVICE} PYTHONPATH=$(pwd) python src_llm/FRAIN.py    --iid=${IID} --byzantines=5  --score_byzantines=5  --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --drift=21 --interpol=slerp --adaptive=hinge --adaptive_a=10.0 --adaptive_b=4.0 --adaptive_c=16.0


    done
done
