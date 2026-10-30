# Network Intrusion Detection Under Distribution Shift

**A two-stage classifier for network traffic whose test set doesn't look like its training set.**

**Author:** Venkata Naga Rohit Pogula · [Kaggle profile](https://www.kaggle.com/rohitpogula)

This repository contains my solution to the **Network Intrusion Detection ML Challenge**, the GDG on Campus AI/ML volunteer recruitment challenge hosted on Kaggle. It includes the full path from first exploration to the final model, along with the experiments that didn't work and why.

| | |
|---|---|
| **Task** | Classify network flows as Availability, Benign, Reconnaissance or Spoofing |
| **Metric** | Macro-averaged F1 (every class counts equally) |
| **Final public score** | **0.84596** |
| **Final model** | Two-stage LightGBM: an Availability detector, then pairwise specialists for the other three classes |

---

## Contents

1. [The problem](#1-the-problem)
2. [Exploring the data](#2-exploring-the-data)
3. [Building a validation scheme I could trust](#3-building-a-validation-scheme-i-could-trust)
4. [From one model to two stages](#4-from-one-model-to-two-stages)
5. [Improving stage 2](#5-improving-stage-2)
6. [What didn't work](#6-what-didnt-work)
7. [Results](#7-results)
8. [Lessons learned](#8-lessons-learned)
9. [Repository structure](#9-repository-structure)
10. [How to run](#10-how-to-run)

---

## 1. The problem

The competition provides 84,713 labelled training flows and 33,161 unlabelled test flows. Each flow is described by 82 features: 80 numeric traffic statistics (packet counts, payload sizes, header sizes, TCP flag counts, inter-arrival times, rates, window sizes) plus two categorical fields, `proto` and `service`.

The organisers built in a deliberate twist. The test set contains **traffic behaviour profiles that are absent from the training data**, and the public and private leaderboards each hold different hidden profiles. A model that scores well on a random validation split can still fall apart on the leaderboard. That shaped every decision I made.

## 2. Exploring the data

### A heavily imbalanced dataset, dominated by a single attack

The class balance was the first thing that stood out:

| Class | Training rows | Share |
|---|---:|---:|
| Availability | 80,072 | 94.5% |
| Spoofing | 2,323 | 2.7% |
| Benign | 1,583 | 1.9% |
| Reconnaissance | 735 | 0.9% |

Looking closer, 79,698 of the 80,072 Availability rows share almost the same fingerprint: a single 120-byte forward packet to port 21 with identical header and window sizes. In other words, nearly all of the "Availability" class is **one SYN-flood attack**, repeated tens of thousands of times.

The data itself was clean. There were no missing values, no infinities and no duplicate rows. One column (`fwd_URG_flag_count`) is constant in train. A few categories appear only in train (`icmp` in `proto`; `dhcp`, `irc`, `ntp` and `radius` in `service`).

### Ports are almost labels, and that is a trap

Response ports map almost one-to-one onto classes in train. Port 21 is the flood, port 1883 (MQTT) is entirely Benign, and so on. That looks like a gift, but the test set has **1,733 distinct response ports versus 308 in train**, and only about 82% of test rows use a port seen in training. A model that memorises ports will memorise the training profiles, which is exactly what this competition punishes.

### Train and test are almost perfectly separable

To measure the shift directly, I trained an *adversarial* classifier whose only job is to tell training rows from test rows. It reached an **AUC of 1.0000**, which means train and test are completely distinguishable. Removing the port columns barely changed that (0.9997), so the shift isn't just about ports.

Per-feature comparisons showed where the difference really lives:

| Feature | Train median | Test median | KS statistic |
|---|---:|---:|---:|
| `payload_bytes_per_second` | 31,457,280 | 0 | 0.94 |
| `flow_pkts_per_sec` | 524,288 | 0.36 | 0.88 |
| `fwd_pkts_per_sec` / `bwd_pkts_per_sec` | 262,144 | 0.13-0.23 | 0.88 |

The shift is concentrated in **timing and rate features** (durations, inter-arrival times, packets and bytes per second) and in **response-side features** (whether the target replied, and RST/ACK counts).

### The key observation

The test set has 10,396 flows to port 21. When I compared them with the training flood, every **forward-packet** feature matched exactly: same packet size, same headers, same window size, a Kolmogorov-Smirnov statistic of 0. But their **timing and reply** features were completely different: zero duration, often no reply at all.

So the test set contains the same SYN-flood attack captured under different conditions. A model that leans on timing features doesn't recognise it. My first baseline did exactly that. It labelled about 9,100 of those port-21 test flows as Benign and predicted only 2.9% of the test set as Availability.

## 3. Building a validation scheme I could trust

A random K-fold split scores around 0.99 here and tells you nothing, because every validation fold contains the same profiles as the training folds. I needed validation that imitates "unseen behaviour".

**Cluster-held-out cross-validation.** Within each class, I clustered the training rows on their behaviour features (log-scaled, standardised and reduced with PCA, with ports excluded). Each cluster acts as a pseudo "behaviour profile". I then held out **whole clusters** at a time, so every validation fold contains profiles the model has never trained on.

Two refinements mattered:

- **Grouping by port doesn't work.** My first attempt held out whole ports instead of clusters. Because port is so tied to the label, entire classes vanished from the training folds and Availability recall dropped to zero. It produced a meaningless score of 0.048.
- **Repeated splits have to be genuinely different.** In one round of tuning I discovered that my "repeated" splits were identical, because the splitter ignored the random seed. That inflated an apparent gain, and I threw the result away. In the fixed version, both the clustering and the cluster-to-fold assignment change with the seed, and decisions are confirmed on **fresh splits never used during the search**.

My rule for accepting any change: it had to improve the cross-validated macro F1 by **at least 0.01 on average and win on at least 4 of 5 fresh splits**. Anything smaller I treated as noise.

## 4. From one model to two stages

### A single model hits a wall

With a single LightGBM classifier, I compared several feature sets under the cluster-held-out validation:

| Feature set | Grouped-CV macro F1 |
|---|---:|
| All features | 0.725 |
| No ports | 0.706 |
| No ports, no timing/rate | 0.520 |
| No ports, no timing/rate/response | 0.532 |

This exposed a real conflict. The timing features are **needed** to separate Benign, Reconnaissance and Spoofing, since removing them collapses the score. But they are also exactly what **breaks** the Availability decision on the test set's port-21 flood. No single feature set could serve both purposes.

### Splitting the problem

The solution was to give each sub-problem the features it can trust:

**Stage 1: Availability vs everything else.** This stage uses only stable forward-packet features, with no ports, timing, rate or response features. It works because the flood is defined by what the attacker *sends*, not by how the target responds. In cross-validation it reached an Availability F1 of 0.997, and on the test set it flags 31.3% of rows, essentially the entire port-21 flood.

**Stage 2: Benign vs Reconnaissance vs Spoofing.** This stage is trained only on the 4,641 non-Availability training rows, with all features, because here the timing information carries real signal.

The leaderboard confirmed the reasoning. The two-stage model jumped from the low 0.7s to **0.845**. A controlled ablation showed why stage 2 must keep the timing features: removing them dropped the score to **0.729**, and removing only the four rate features dropped it to **0.822**.

## 5. Improving stage 2

### Tuning with honest validation

I ran a random search over LightGBM settings for stage 2, scored with repeated cluster-held-out CV, and kept the average of the **top three settings**. On fresh splits this improved stage-2 macro F1 from 0.772 to 0.783, winning on 4 of 5 splits. On the leaderboard it was a small but consistent step (0.84543).

### Pairwise specialists: the final model

A single 3-class model has to compromise. The features that separate Benign from Spoofing (packet symmetry, payload sizes) aren't the same ones that separate Benign from Reconnaissance (reply behaviour, packet counts). So I replaced it with **three binary specialists**, one per pair of classes:

- Benign vs Reconnaissance
- Benign vs Spoofing
- Reconnaissance vs Spoofing

Each specialist trains only on the rows of its two classes and can focus on its own boundary. Their pairwise probabilities are combined into 3-class probabilities with **pairwise coupling** (Price's method).

This was the clearest improvement of the project:

| Stage 2 | Macro F1 | Benign | Reconnaissance | Spoofing |
|---|---:|---:|---:|---:|
| Single tuned 3-class model | 0.767 | 0.782 | 0.767 | 0.753 |
| **Pairwise specialists** | **0.801** | **0.800** | **0.823** | **0.781** |

It won on all five fresh splits, and Reconnaissance, the hardest class, gained the most. On the leaderboard it became my best submission at **0.84596**. The leaderboard gain is smaller than the cross-validation gain because most test flows are clear-cut cases that both approaches already agree on; the specialists change about 225 test predictions.

## 6. What didn't work

I'm including the failures because they shaped the final model as much as the successes did. Each of these was rejected by cross-validation rather than by the leaderboard.

| Idea | Result | Why I think it failed |
|---|---|---|
| Engineered ratio features (bytes per packet, flag fractions, IAT coefficient of variation) | 0.765 vs 0.767 | Gradient-boosted trees already capture most of these interactions |
| Adding ExtraTrees to the stage-2 blend | Dropped from about 0.64 to 0.47-0.52 | Underfits on only about 4,600 minority-class rows |
| Pseudo-labeling (retraining on confident test predictions) | 0.731 vs 0.754, unstable | On one split the model reinforced its own mistakes |
| Tuning each pairwise specialist separately | 0.780 vs 0.794 | Each pair's log loss improved, but the combined prediction got worse |
| Multi-family ensemble (LightGBM + XGBoost + CatBoost) | Leaderboard 0.84544, about level | Similar tree models tend to make the same mistakes |
| Lowering the stage-1 decision threshold | No change at all | Stage 1 is near-certain on every test row |

## 7. Results

Public leaderboard scores (about 40% of the test set):

| Submission | Public macro F1 |
|---|---:|
| Two-stage model, stage 2 without timing/rate features (ablation) | 0.72928 |
| Two-stage model, stage 2 without the four rate features (ablation) | 0.82231 |
| Two-stage model, stage 2 without ports | 0.84253 |
| Two-stage model, stage 2 with all features | 0.84538 |
| Tuned stage 2 (average of three LightGBM settings) | 0.84543 |
| Multi-family ensemble, calibrated | 0.84544 |
| **Pairwise specialists (final)** | **0.84596** |

The final model predicts the test set as 31.3% Availability, 31.4% Benign, 20.9% Reconnaissance and 16.4% Spoofing.

## 8. Lessons learned

- **Validation design matters more than model choice.** Once I had validation that imitated the real shift, most decisions became obvious. Without it, I would have been tuning towards a meaningless 0.99.
- **Measure the shift before modelling.** The adversarial check pointed straight at the timing features, and that led directly to the two-stage design.
- **The same feature can be essential in one sub-problem and harmful in another.** Splitting the task let each stage use the features that hold up for its own question.
- **Small leaderboard differences are mostly noise.** With a public set of about 13,000 rows, a gap of 0.002 is a handful of predictions. I made decisions with cross-validation and treated the leaderboard as a final check.
- **Write down the failures.** Most of the ideas I tried didn't help, and knowing why was what kept the final model simple.

## 9. Repository structure

```
notebooks/
  01_eda_baseline.ipynb                  exploration, the train-vs-test shift check, a first baseline
  02_two_stage_model.ipynb               the two-stage design and the feature-set comparisons
  03_tuned_stage2.ipynb                  stage-2 tuning with repeated cluster-held-out CV
  04_pairwise_specialists_FINAL.ipynb    the final model; writes the final submission
  experiments/
    multi_family_ensemble.ipynb          LightGBM + XGBoost + CatBoost stage 2
    pseudo_labeling.ipynb                self-training on confident test predictions
    per_pair_tuned_specialists.ipynb     separately tuned pairwise specialists
data/                                    
requirements.txt
```

## 10. How to run

**On Kaggle (easiest).** Open the competition page, choose **Code → New Notebook**, then **File → Import Notebook** and select any notebook from `notebooks/`. Click **Run All**. The data is found automatically under `/kaggle/input`.

**Locally.**

```bash
pip install -r requirements.txt
# Download train.csv, test.csv and sample_submission.csv from the competition's Data tab into data/
jupyter notebook notebooks/04_pairwise_specialists_FINAL.ipynb
```

The final notebook takes about three to five minutes on a laptop CPU and writes `submission_v10_specialists.csv` next to the notebook.

---

### Data and acknowledgements

The competition data isn't included in this repository; it is available from the competition page and subject to the competition rules. The dataset was created by B. S. and Rohini Nagapadma (The National Institute of Engineering, Mysuru) and is licensed under CC BY 4.0.

This project was developed with the help of an AI coding assistant, which the competition rules permit. The analysis, the validation design and every modelling decision described here were checked against the cross-validation results above.
