# SHy (Demo/Subset Implementation)

Implementation of:

**Self-Explaining Hypergraph Neural Networks for Diagnosis Prediction (SHy)**

이 저장소는 SHy 모델을 **MIMIC-III / MIMIC-IV demo 또는 축소 데이터 환경**에서 실행하고,  
전처리-학습-평가 파이프라인이 실제로 동작하는지 검증한 구현입니다.


---

## 1. Project Overview

SHy는 환자의 과거 방문 기록을 단순 시계열이 아니라 **하이퍼그래프(hypergraph)** 로 표현하고,  
여기서 **temporal phenotype**을 추출하여 다음 방문의 진단 코드를 예측하는 모델입니다.

핵심 아이디어는 다음과 같습니다.

- 각 질병 코드를 노드(node)로 둡니다.
- 각 방문(visit)을 하이퍼엣지(hyperedge)로 둡니다.
- ICD hierarchy를 반영한 질병 임베딩을 초기화합니다.
- 환자별 personalized disease embedding을 학습합니다.
- 여러 개의 temporal phenotype을 추출합니다.
- phenotype를 기반으로 다음 방문 진단을 예측합니다.

SHy는 **환자별 phenotype 자체를 설명 단위로 사용하는 self-explaining 모델**입니다.

---

## 2. Project Structure

```text
SHy/
├─ src/
│  ├─ main.py
│  ├─ model.py
│  ├─ training.py
│  ├─ dataset.py
│  ├─ iii_preprocessing.ipynb
│  └─ iv_preprocessing.ipynb
├─ data/
│  ├─ RAW/
│  │  ├─ MIMIC_III/
│  │  └─ MIMIC_IV/
│  ├─ MIMIC_III/
│  └─ MIMIC_IV/
├─ saved_models/
├─ training_logs/
├─ run_mimic3_50.ps1
├─ run_mimic4_50.ps1
├─ aggregate_results.py
└─ README.md
```

설명:
- `src/`: 모델 코드 및 전처리 notebook
- `data/RAW/`: 원본 데이터 파일 위치
- `data/MIMIC_III`, `data/MIMIC_IV`: 전처리 결과 저장 위치
- `training_logs/`: metric, loss, plot 결과 저장 위치
- `saved_models/`: 모델 저장 위치
- `run_*.ps1`: 반복 실행 스크립트
- `aggregate_results.py`: 여러 실행 결과 평균/표준편차 집계 스크립트

---

## 3. Environment Setup

### 3.1 Conda environment 생성

```bash
conda create -n shy python=3.9 -y
conda activate shy
python -m pip install --upgrade pip
```

### 3.2 패키지 설치

```bash
pip install torch==1.13.1 pyro-ppl==1.8.4
pip install torch-geometric==2.2.0
pip install numpy==1.26.4 pandas scipy scikit-learn matplotlib jupyter notebook
```

주의:
- 본 구현은 **CPU 환경 기준**으로 먼저 안정화했습니다.
- GPU 환경은 추후 코드 개선 예정입니다.

---

## 4. Data Preparation

### 4.1 MIMIC-III

다음 파일을 `data/RAW/MIMIC_III/` 아래에 둡니다.

- `ADMISSIONS.csv`
- `DIAGNOSES_ICD.csv`
- `D_ICD_DIAGNOSES.csv`
- `icd9.txt`

전처리 notebook:

```text
src/iii_preprocessing.ipynb
```

### 4.2 MIMIC-IV

다음 파일을 `data/RAW/MIMIC_IV/` 아래에 둡니다.

- `admissions.csv`
- `diagnoses_icd.csv`
- `d_icd_diagnoses.csv`
- `icd9.txt`
- `dump_list_icd9.pkl`

참고:
- `.csv.gz` 형식으로 받은 경우 먼저 `.csv`로 압축을 해제한 뒤 사용했습니다.

전처리 notebook:

```text
src/iv_preprocessing.ipynb
```

---

## 5. Preprocessing

전처리는 다음 notebook을 사용합니다.

- `src/iii_preprocessing.ipynb`
- `src/iv_preprocessing.ipynb`

전처리 단계에서는 다음 작업을 수행합니다.

- admission 파싱
- diagnosis 파싱
- 환자별 방문 정리
- diagnosis code 인코딩
- visit length 생성
- ICD hierarchy level 생성
- binary visit-code matrix 생성
- train/test split 생성
- `.pkl`, `.npy` 저장

---

## 6. Preprocessing Fixes

원본 전처리 코드는 full dataset을 전제로 되어 있어,  
demo/subset 환경에서는 그대로 실행되지 않았습니다.

따라서 본 프로젝트에서는 다음과 같은 수정을 반영했습니다.

### 6.1 Deprecated NumPy aliases 수정
다음 표현을 수정했습니다.

- `np.int` → `int`
- `np.str` → `str`

### 6.2 CSV 컬럼명 처리 수정
데이터 파일에 따라 컬럼명이 대문자/소문자로 달라,  
실제 파일 구조에 맞게 컬럼명을 정리했습니다.

### 6.3 Split 로직 수정
원본 split 로직은 full dataset 기준 하드코딩 값이 포함되어 있어,  
demo/subset 환경에서는 test set이 비는 문제가 발생했습니다.

따라서:
- 최소 1개 이상의 test sample이 존재하도록
- 작은 데이터에서도 split이 가능하도록

split 방식을 수정했습니다.

### 6.4 ICD hierarchy level fallback 수정
원본 전처리의 `generate_code_levels()`는  
매핑되지 않는 코드를 `[0, 0, 0, 0]`으로 처리하고 있었습니다.

이 방식은 이후 임베딩 단계에서 문제를 일으킬 수 있으므로,  
본 구현에서는 **unknown code도 양수 level을 가지는 별도 bucket**으로 처리하도록 수정했습니다.

### 6.5 Relative Path 정리
전처리 결과 저장 경로를 `../data/...` 기준으로 정리하여,  
학습 코드가 기대하는 파일 위치와 일치하도록 맞췄습니다.

중요:
이 수정은 원 논문 전처리의 strict reproduction이 아니라,  
**축소 데이터 환경에서 실험이 가능하도록 안정화한 전처리**입니다.

---

## 7. Training

본 프로젝트에서는 **50 epoch 기준 실험**을 사용했습니다.
**50 epoch 기준 실험**을 한 이유는 데이터셋이 demo버전이기 때문에 epoch가 커지면 과적합이 발생하기 때문입니다.


### 7.1 MIMIC-III

`src/`에서 다음 명령어를 실행합니다.

```bash
python -u main.py --dataset_name MIMIC_III --num_epoch 50 --batch_size 4 --temperature 1.0 1.0 1.0 1.0 1.0 --add_ratio 0.2 0.2 0.2 0.2 0.2 --loss_weight 1.0 0.003 0.00025 0.0 0.04
```

### 7.2 MIMIC-IV

`src/`에서 다음 명령어를 실행합니다.

```bash
python -u main.py --dataset_name MIMIC_IV --num_epoch 50 --batch_size 4 --temperature 1.0 1.0 1.0 1.0 1.0 --add_ratio 0.2 0.2 0.2 0.2 0.2 --loss_weight 1.0 0.003 0.00025 0.0 0.04
```

설명:
- `batch_size=4`는 demo/subset 환경에서의 안정성을 고려한 값입니다.
- 본 프로젝트는 50 epoch 기준 실험을 사용했습니다.

결과:
- 모델 체크포인트는 `saved_models/`에 저장됩니다.
- metric / loss / plot 결과는 `training_logs/`에 저장됩니다.

---

## 8. Repeated Runs (5 Runs)

논문처럼 여러 번 실행한 결과를 평균내기 위해,  
본 프로젝트에서는 각 실험을 **서로 다른 random seed로 5회 반복 실행**했습니다.

사용한 seed:
- 3407
- 3408
- 3409
- 3410
- 3411

---

## 9. Run Scripts

### 9.1 MIMIC-III, 50 epoch

```powershell
$seeds = @(3407, 3408, 3409, 3410, 3411)

conda activate shy
Set-Location C:\Users\dhkdd\Desktop\shy\SHy\src

foreach ($s in $seeds) {
    python -u main.py `
        --dataset_name MIMIC_III `
        --seed $s `
        --num_epoch 50 `
        --batch_size 4 `
        --temperature 1.0 1.0 1.0 1.0 1.0 `
        --add_ratio 0.2 0.2 0.2 0.2 0.2 `
        --loss_weight 1.0 0.003 0.00025 0.0 0.04
}
```

### 9.2 MIMIC-IV, 50 epoch

```powershell
$seeds = @(3407, 3408, 3409, 3410, 3411)

conda activate shy
Set-Location C:\Users\dhkdd\Desktop\shy\SHy\src

foreach ($s in $seeds) {
    python -u main.py `
        --dataset_name MIMIC_IV `
        --seed $s `
        --num_epoch 50 `
        --batch_size 4 `
        --temperature 1.0 1.0 1.0 1.0 1.0 `
        --add_ratio 0.2 0.2 0.2 0.2 0.2 `
        --loss_weight 1.0 0.003 0.00025 0.0 0.04
}
```

---

## 10. Result Aggregation

각 실행 결과는 `training_logs/` 아래에 개별 run 폴더로 저장됩니다.

주요 저장 파일:
- `r2_list.pkl`
- `r4_list.pkl`
- `n2_list.pkl`
- `n4_list.pkl`
- `train_average_loss_per_epoch.pkl`
- `test_loss_per_epoch.pkl`
- `prediction_loss_per_epoch.pkl`

집계 스크립트:

```bash
python aggregate_results.py
```

이 스크립트는 다음을 자동으로 분리하여 집계합니다.

- `MIMIC_III`
- `MIMIC_IV`
- epoch 길이별 그룹

집계 항목:
- best Recall@10 / Recall@20 / nDCG@10 / nDCG@20
- final Recall@10 / Recall@20 / nDCG@10 / nDCG@20
- best test loss
- final train/test/prediction loss
- 평균 ± 표준편차

---

## 11. Main Results

demo 데이터 대표 결과는 **50 epoch 기준 평균 결과**를 사용했습니다.

| Dataset | Setting | Recall@10 | Recall@20 | nDCG@10 | nDCG@20 | Test Loss |
|--------|---------|-----------|-----------|---------|---------|-----------|
| MIMIC-III | Paper (SHy) | 0.2775 | 0.3831 | 0.4135 | 0.4088 | - |
| MIMIC-III | Ours (50 epoch, 5 runs) | 0.2199 ± 0.0413 | 0.2844 ± 0.0399 | 0.2765 ± 0.0642 | 0.2661 ± 0.0512 | 0.2657 ± 0.0025 |
| MIMIC-IV | Paper (SHy) | 0.3344 | 0.4270 | 0.4236 | 0.4239 | - |
| MIMIC-IV | Ours (50 epoch, 5 runs) | 0.2276 ± 0.0253 | 0.2823 ± 0.0270 | 0.4244 ± 0.0276 | 0.3926 ± 0.0241 | 0.2772 ± 0.0017 |

---  

### Metric 설명

- **Recall@10 / Recall@20**: 상위 10개 또는 20개 예측 안에 실제 정답 진단이 얼마나 포함되었는지를 나타내는 지표
- **nDCG@10 / nDCG@20**: 상위 10개 또는 20개 예측에서 실제 정답 진단이 얼마나 높은 순위에 배치되었는지를 반영하는 지표
- **Test Loss**: 테스트 데이터에서의 전체 예측 오차를 나타내는 값으로, 일반적으로 낮을수록 좋음


## 12. Visualization

학습 후 생성된 시각화 이미지는 다음과 같이 해석할 수 있습니다.

아래의 이미지들을  **50 epoch 기준**으로 MIMIC-IV를 돌린 결과값입니다.

![Total Loss](./training_logs/04_09_2026M13_53_53__3411__MIMIC_IV/total_loss_plot.svg)
### `total_loss_plot.svg`
- 빨간색은 **training loss**, 파란색은 **test loss**를 나타낸다.
- 본 실험에서는 training loss가 epoch가 진행될수록 지속적으로 감소하였다.
- 반면 test loss는 초반에 소폭 감소한 뒤, 중후반으로 갈수록 다시 증가하는 경향을 보였다.
- 이는 모델이 학습 데이터에는 점점 더 잘 적합되지만, 테스트 데이터에 대한 일반화 성능은 후반부에 오히려 악화될 수 있음을 의미한다.
- 즉, 이 그래프는 **small-data 환경에서 과적합이 발생하고 있음을 보여주는 대표적인 신호**로 해석할 수 있다.

<br>
<br>

![Total Loss](./training_logs/04_09_2026M13_53_53__3411__MIMIC_IV/prediction_loss_plot.svg)
### `prediction_loss_plot.svg`
- 이 그래프는 테스트 데이터 기준 **prediction loss**의 변화만 따로 시각화한 것이다.
- 곡선을 보면 초반 몇 epoch 동안 loss가 감소하여, 대략 **10~15 epoch 부근에서 가장 낮은 값**을 형성한다.
- 이후에는 prediction loss가 다시 점진적으로 증가한다.
- 이는 장기 학습이 항상 예측 성능 향상으로 이어지는 것이 아니라,  
  오히려 **초반 학습 구간이 더 좋은 일반화 성능을 보일 수 있음**을 의미한다.
- 따라서 본 실험에서는 단순히 epoch를 많이 늘리는 것보다,  
  **적절한 epoch 지점을 선택하는 것이 더 중요하다**는 점을 확인할 수 있었다.

<br>
<br>
  
![Total Loss](./training_logs/04_09_2026M13_53_53__3411__MIMIC_IV/recall_plot.svg)
### `recall_plot.svg`
- 이 그래프는 epoch에 따른 **Recall@k**의 변화를 보여준다.
- 전체적으로 보면 recall 값은 학습 초반보다 후반으로 갈수록 상승하는 추세를 보인다.
- 즉, 모델이 학습되면서 정답 진단 코드를 상위 예측 목록 안에 포함시키는 능력이 개선되었다고 볼 수 있다.
- 다만 곡선의 진동 폭이 비교적 큰 편인데, 이는 현재 실험이 **demo/subset 기반의 작은 데이터셋**으로 수행되었기 때문에 metric 변동성이 크게 나타난 결과로 해석할 수 있다.
- 따라서 recall 상승은 긍정적이지만, **작은 데이터에서의 불안정성도 함께 고려**해야 한다.

<br>
<br>


![Total Loss](./training_logs/04_09_2026M13_53_53__3411__MIMIC_IV/ndcg_plot.svg)
### `ndcg_plot.svg`
- 이 그래프는 epoch에 따른 **nDCG@k**의 변화를 보여준다.
- nDCG는 단순히 정답을 맞췄는지뿐 아니라,  
  그 정답을 **얼마나 높은 순위에 배치했는지**까지 반영하는 지표이다.
- 그래프를 보면 nDCG는 전체적으로 꾸준한 상승 추세를 보이며,  
  이는 학습이 진행될수록 모델이 정답 진단 코드를 더 상위에 배치하는 방향으로 개선되고 있음을 의미한다.
- Recall 그래프와 비교하면, nDCG는 상대적으로 더 안정적인 상승 패턴을 보인다.
- 따라서 본 실험에서는 SHy가 작은 데이터 환경에서도  
  **상위 진단 후보의 정렬 품질(ranking quality)** 은 일정 수준 학습하고 있음을 확인할 수 있었다.


## 13. Interpretation of Our Results


- 논문과 완전히 동일한 수치를 재현한 것은 아니지만,
- SHy의 핵심 구조와 실행 흐름은 정상적으로 검증했습니다.

특히 MIMIC-III와 MIMIC-IV 모두에서:
- 전처리 성공
- 모델 생성 성공
- 학습 루프 성공
- metric 계산 성공

을 확인했습니다.

---

## 14. Limitations

이 프로젝트의 한계는 다음과 같습니다.

1. full MIMIC 데이터가 아니라 demo/subset 기반입니다.
2. 전처리 단계에서 small-data 환경에 맞춘 보정이 포함되었습니다.
3. 위에서 말했듯 일부 metric이 높게 보이더라도 작은 데이터 분산의 영향일 수 있습니다.


## 15. Conclusion

본 프로젝트는 SHy를 demo/subset 환경에서 실행 가능하게 만들고,  
MIMIC-III와 MIMIC-IV 모두에 대해 전처리-학습-평가 파이프라인을 검증한 구현입니다.

핵심적으로 확인한 사항은 다음과 같습니다.

- SHy의 hypergraph 기반 구조는 축소 환경에서도 동작한다.
- temporal phenotype extraction과 ranking metric 계산이 정상적으로 수행된다.
- 다만 full-data 논문 재현은 demo의 한계성으로 인해 구현을 실패하였다.
- 따라서 본 프로젝트는 **논문 구조 이해 및 축소 환경 검증용 구현**으로 보는 것이 적절하다.
