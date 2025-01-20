#!/bin/sh

data=ctu
echo Data: $data

for botnet in neris rbot virut menti murlo
do
    echo Botnet: $botnet

    echo Script: preprocessing.py
    python scripts/$data/preprocessing.py --benign-data data/$data/raw/benign.csv --malicious-data data/$data/raw/malicious/$botnet.csv --train-data data/$data/processed/train/$botnet.csv --test-data data/$data/processed/test/$botnet.csv

    echo Script: graphsage.py
    python scripts/$data/graphsage.py --train-data data/$data/processed/train/$botnet.csv --model models/$data/graphsage/$botnet.pth

    echo Script: dummyexplainer.py
    python scripts/$data/dummyexplainer.py --test-data data/$data/processed/test/$botnet.csv --model models/$data/graphsage/$botnet.pth --scores scores/$data/graphsage/dummyexplainer/$botnet.csv

    echo Script: gnnexplainer.py
    python scripts/$data/gnnexplainer.py --test-data data/$data/processed/test/$botnet.csv --model models/$data/graphsage/$botnet.pth --scores scores/$data/graphsage/gnnexplainer/$botnet.csv

    echo Script: graphmaskexplainer.py
    python scripts/$data/graphmaskexplainer.py --test-data data/$data/processed/test/$botnet.csv --model models/$data/graphsage/$botnet.pth --scores scores/$data/graphsage/graphmaskexplainer/$botnet.csv

    echo Script: integratedgradients.py
    python scripts/$data/integratedgradients.py --test-data data/$data/processed/test/$botnet.csv --model models/$data/graphsage/$botnet.pth --scores scores/$data/graphsage/integratedgradients/$botnet.csv

    echo Script: saliency.py
    python scripts/$data/saliency.py --test-data data/$data/processed/test/$botnet.csv --model models/$data/graphsage/$botnet.pth --scores scores/$data/graphsage/saliency/$botnet.csv
done

data=ton
echo Data: $data

for attack in backdoor ddos dos injection password ransomware scanning xss
do
    echo Attack: $attack

    echo Script: preprocessing.py
    python scripts/$data/preprocessing.py --benign-data data/$data/raw/benign.csv --malicious-data data/$data/raw/malicious/$attack.csv --train-data data/$data/processed/train/$attack.csv --test-data data/$data/processed/test/$attack.csv

    echo Script: graphsage.py
    python scripts/$data/graphsage.py --train-data data/$data/processed/train/$attack.csv --model models/$data/graphsage/$attack.pth

    echo Script: dummyexplainer.py
    python scripts/$data/dummyexplainer.py --test-data data/$data/processed/test/$attack.csv --model models/$data/graphsage/$attack.pth --scores scores/$data/graphsage/dummyexplainer/$attack.csv

    echo Script: gnnexplainer.py
    python scripts/$data/gnnexplainer.py --test-data data/$data/processed/test/$attack.csv --model models/$data/graphsage/$attack.pth --scores scores/$data/graphsage/gnnexplainer/$attack.csv

    echo Script: graphmaskexplainer.py
    python scripts/$data/graphmaskexplainer.py --test-data data/$data/processed/test/$attack.csv --model models/$data/graphsage/$attack.pth --scores scores/$data/graphsage/graphmaskexplainer/$attack.csv

    echo Script: integratedgradients.py
    python scripts/$data/integratedgradients.py --test-data data/$data/processed/test/$attack.csv --model models/$data/graphsage/$attack.pth --scores scores/$data/graphsage/integratedgradients/$attack.csv

    echo Script: saliency.py
    python scripts/$data/saliency.py --test-data data/$data/processed/test/$attack.csv --model models/$data/graphsage/$attack.pth --scores scores/$data/graphsage/saliency/$attack.csv
done