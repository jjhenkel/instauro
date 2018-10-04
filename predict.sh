#!/bin/bash

echo "[instauro] Running featurizer (this can be SLOW)..."
python3 /app/src/featurizer.py /data/Tasks > /tmp/features.txt
echo "[instauro] Ranking use pre-trained model..."
java -jar /app/lib/RankLib-2.8.jar -load /app/models/pDS-all.model -rank /tmp/features.txt -score /tmp/scores.txt
echo "[instauro] Making predictions (this can also take a while)..."
python3 /app/src/guesser.py /tmp/scores.txt /data/Tasks | python3 /app/src/evaluate.py -d /data
echo "[instauro] Done!"
