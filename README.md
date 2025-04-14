# MOUM Project: Multi-Omics Based Drug Response Prediction

## Project Overview

본 프로젝트는 multi-omics 데이터를 활용한 약물 반응 예측 모델 개발을 목표로 합니다.

## Core Components

### 1. Multi-omics Data Integration Method (윤진)
- 다양한 omics 데이터의 통합 방법론 개발
- 데이터 통합을 통한 세포 특성 이해 향상

### 2. Cell-type Specific Foundation Model (경덕)
- 세포 타입별 특화된 기초 모델 개발
- 세포 특성을 반영한 데이터 표현 학습

### 3. Drug Response Prediction (대성)
- 약물 반응 예측 모델 개발
- 주요 구성 요소:
  - Cell type specific omics data encoding
  - Drug representation encoding

## Development Pipeline

### Step 1: Omics Data Integration Model
1. 필요한 omics 데이터 선정
2. 데이터 통합 모델 개발
3. 간단한 downstream task를 통한 성능 검증
4. Cell type specific foundation model 적용 검토

### Step 2: Drug Response Predictor 개발
1. Drug embedding 개발
2. Multiomics 데이터와 drug embedding 통합
3. 최종 Drug reaction predictor 구현

---
### Data Sources


**공개 데이터베이스**
1. Cancer Cell Line Encyclopedia (CCLE)
제공 데이터: 암세포주의 유전체, 전사체, 단백질체, 약물 반응 데이터
특징: 1,000개 이상의 세포주에 대한 다양한 오믹스 데이터
URL: https://portals.broadinstitute.org/ccle
2. The Cancer Genome Atlas (TCGA)
제공 데이터: 환자 샘플의 genomic, transcriptomic, epigenomic 데이터
특징: 다양한 암 유형에 대한 임상 정보 포함
URL: https://portal.gdc.cancer.gov/
3. Genomics of Drug Sensitivity in Cancer (GDSC)
제공 데이터: 암세포주의 약물 감수성 데이터, 유전자 변이 정보
특징: 다양한 표적 약물에 대한 반응 프로파일
URL: https://www.cancerrxgene.org/
4. GEO (Gene Expression Omnibus)
제공 데이터: 유전자 발현, 메틸화, ChIP-seq 등의 데이터
특징: 다양한 실험 디자인의 오믹스 데이터 저장소
URL: https://www.ncbi.nlm.nih.gov/geo/
5. PharmacoDB
제공 데이터: 여러 약물 스크리닝 연구의 통합 데이터
특징: CCLE, GDSC, CTRPv2 등의 데이터를 통합
URL: https://pharmacodb.pmgenomics.ca/
통합할 오믹스 데이터 유형
유전체학(Genomics): SNP, CNV, 돌연변이
전사체학(Transcriptomics): RNA-seq, 마이크로어레이
단백질체학(Proteomics): 단백질 발현, 인산화
에피지놈학(Epigenomics): DNA 메틸화, 히스톤 수정
대사체학(Metabolomics): 대사물질 프로파일
데이터 통합 접근 방식
초기 단계 통합(Early integration)
다양한 오믹스 데이터를 하나의 특성 벡터로 통합
간단하지만 오믹스 데이터 간의 복잡한 관계를 포착하기 어려움
중간 단계 통합(Intermediate integration)
각 오믹스 데이터를 개별적으로 처리한 후 결합
예: Multi-modal neural networks, 앙상블 방법
후기 단계 통합(Late integration)
각 오믹스 데이터로 개별 모델 구축 후 결과 통합
예: Stacking, 투표 전략

---

## Todo List

### 진행중
- [ ] Multi-omics Data Integration Method 개발
- [ ] Cell-type Specific Foundation Model 구현
- [ ] Drug Response Prediction 모델 개발

### 완료

### 보류

---

## Original Project Template Information

### 쿠키커터 프로젝트 소개

쿠키커터는 프로젝트 템플릿(쿠키커터)에서 신속하게 프로젝트를 생성할 수 있는 커맨드라인 유틸리티입니다. Python 패키지 프로젝트 생성 등에 이상적입니다.

### 설치 방법

```bash
# pipx 사용 권장
pipx install cookiecutter

# pipx를 사용할 수 없는 경우
python -m pip install --user cookiecutter
```
