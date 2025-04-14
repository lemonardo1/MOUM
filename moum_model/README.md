# MOUM Project: Multi-Omics Based Drug Response Prediction

## Project Overview

본 프로젝트는 multi-omics 데이터를 활용한 약물 반응 예측 모델 개발을 목표로 합니다.

## Core Components

### 1. Multi-omics Data Integration Method (GNN-based)
- GNN 기반의 multi-omics 데이터 통합 모델
- 주요 기능:
  - 각 omics 데이터 타입별 인코딩
  - GNN을 통한 omics 간 관계 학습
  - 조건부 생성 모델을 통한 데이터 증강
- 모델 구조:
  - OmicsGenerator: 개별 omics 데이터 생성
  - MultiOmicsGenerator: GNN 기반 통합 생성
  - ConditionalMultiOmicsGenerator: 조건부 생성

### 2. Cell-type Specific Foundation Model
- 세포 타입별 특화된 기초 모델 개발
- 세포 특성을 반영한 데이터 표현 학습

### 3. Drug Response Prediction
- 약물 반응 예측 모델 개발
- 주요 구성 요소:
  - Cell type specific omics data encoding
  - Drug representation encoding

## Required Data

### 1. Multi-omics Data
- Gene Expression (RNA-seq)
  - Format: Gene x Sample matrix
  - Normalization: TPM or FPKM
  - Required features: Gene symbols, Expression values

- DNA Methylation
  - Format: CpG site x Sample matrix
  - Required features: Beta values, CpG site coordinates

- Copy Number Variation
  - Format: Gene x Sample matrix
  - Required features: Log2 ratio values

### 2. Metadata
- Sample Information
  - Cell line identifiers
  - Tissue types
  - Cancer types
  - Clinical information

- Drug Information
  - Drug identifiers
  - Drug structures
  - Target information
  - Mechanism of action

### 3. Drug Response Data
- IC50 values
- AUC values
- Drug sensitivity scores

## Data Sources

### Public Databases
1. Cancer Cell Line Encyclopedia (CCLE)
   - URL: https://portals.broadinstitute.org/ccle
   - Data types: Gene expression, Copy number, Mutation, Drug response

2. Genomics of Drug Sensitivity in Cancer (GDSC)
   - URL: https://www.cancerrxgene.org/
   - Data types: Drug sensitivity, Gene expression, Mutation

3. The Cancer Genome Atlas (TCGA)
   - URL: https://portal.gdc.cancer.gov/
   - Data types: Multi-omics data, Clinical information

## Model Architecture

### 1. GNN-based Multi-omics Integration
```python
class MultiOmicsGenerator:
    - OmicsGenerator: 개별 omics 데이터 생성
    - GNN layers: omics 간 관계 학습
    - Graph construction: Fully connected or predefined adjacency
```

### 2. Conditional Generation
```python
class ConditionalMultiOmicsGenerator:
    - Condition encoder: 조건 정보 인코딩
    - Latent space manipulation: 조건 정보 통합
    - Conditional generation: 조건에 따른 데이터 생성
```

## Development Pipeline

### Step 1: Data Preparation
1. 데이터 다운로드 및 전처리
2. 데이터 정규화 및 통합
3. 훈련/검증/테스트 세트 분할

### Step 2: Model Development
1. GNN 기반 통합 모델 구현
2. 조건부 생성 모델 구현
3. 모델 훈련 및 검증

### Step 3: Evaluation
1. 생성 데이터 품질 평가
2. Downstream task 성능 평가
3. 모델 해석성 분석

## Todo List

### 진행중
- [ ] Multi-omics Data Integration Method 개발
- [ ] Cell-type Specific Foundation Model 구현
- [ ] Drug Response Prediction 모델 개발

### 완료
- [x] GNN 기반 multi-omics 통합 모델 구현
- [x] 조건부 생성 모델 구현

### 보류

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
