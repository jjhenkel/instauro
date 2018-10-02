# Code Repair Competition

## Our Results

These results are achieved by formulating the code repair task as a "Learn to Rank" task. We use standard textual similarity metrics (and a few custom ones) to turn each `<source line, repair>` tuple into a vector of numbers. Then, we train a `LambdaMART` listwise ranking model (using `RankLib`) on one of the 4 avaliable datasets holding out 20% of the dataset to use as a validation set.

For the results below, I've trained the model on 80% of `Dataset2`. Knowing this, our performance on `Dataset2` itself should be taken with a grain of salt, as it was used to train our ranking model.

## Summary Table

|  Dataset |  Current Best Loss (Lower is better) |  Our Loss (Lower is better) |  Our Recall (Higher is better) |  Percent Improvement | Loss with parseability |
|---|---|---|---|---|---|
|   1| 0.10400918447628195 | 0.08760671358259374 | 0.9116977696859354 | 15.8% | 0.08590981165232915 |
|   2| 0.08405196035050681 | 0.06882598704498212 | 0.9304363537808293 | 18.1% | 0.06768505198157662 |
|   3| 0.05824092991175515 | 0.05736877651117715 | 0.9412869639886223 | 1.5% | 0.055376505680605904 |
|   4| 0.0769219481612277 | 0.06536446227362724 | 0.933516226943731 | 15.0% | 0.06484365418823951 |

## Parseability

This column in the table reports performance using the same ranking model with one post-processing pass. The post-processing pass records the "second best" ranked result and if the file does not parse with the selected best patch **and the file parses in its original form** then the second best ranked result is given.

## Notes

The results above **do not leverage** techniques such as attempting to parse the file with the repair applied. I think we could get a very modest bump in performance (at a speed penalty) by applying some of those tricks. The major disadvantage of our approach is speed. It takes 20-30 minutes (on a 64-core workstation) to create the feature vectors that are used for ranking (and for training the ranker). Furthermore, the ranker must be trained which is another 1-2 hours (but this task only needs to be performed once).

Switching from Python to a compiled language for feature extraction would likely boost the throughput of feature generation. I have a Java feature extract that can handle each dataset (using a single thread) in similar time. The problem is that I couldn't find as many pre-written string similarity libraries for Java (so I was lacking some important features).

## Logs

```bash
# Test on dataset 1 (model trained on 80% of dataset 2)
root@velveeta:/mnt/artifacts/code-rep/Baseline# python3 guessPreds.py ../scores-pDS-2-1.txt 1 | python3 evaluate.py -d /mnt/artifacts/code-rep/Datasets/Dataset1
Total files: 4394
Average line error: 0.08760671358259374 (the lower, the better)
Recall@1: 0.9116977696859354 (the higher, the better)

# Test on dataset 2 (model trained on 80% of dataset 2)
root@velveeta:/mnt/artifacts/code-rep/Baseline# python3 guessPreds.py ../scores-pDS-2-2.txt 2 | python3 evaluate.py -d /mnt/artifacts/code-rep/Datasets/Dataset2
Total files: 11069
Average line error: 0.06882598704498212 (the lower, the better)
Recall@1: 0.9304363537808293 (the higher, the better)

# Test on dataset 3 (model trained on 80% of dataset 2)
root@velveeta:/mnt/artifacts/code-rep/Baseline# python3 guessPreds.py ../scores-pDS-2-3.txt 3 | python3 evaluate.py -d /mnt/artifacts/code-rep/Datasets/Dataset3
Total files: 18633
Average line error: 0.05736877651117715 (the lower, the better)
Recall@1: 0.9412869639886223 (the higher, the better)

# Test on dataset 4 (model trained on 80% of dataset 2)
root@velveeta:/mnt/artifacts/code-rep/Baseline# python3 guessPreds.py ../scores-pDS-2-4.txt 4 | python3 evaluate.py -d /mnt/artifacts/code-rep/Datasets/Dataset4
Total files: 17132
Average line error: 0.06536446227362724 (the lower, the better)
Recall@1: 0.933516226943731 (the higher, the better)
```
