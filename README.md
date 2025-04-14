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

## Original Project Template Information

### 쿠키커터 프로젝트 소개

쿠키커터는 프로젝트 템플릿(쿠키커터)에서 신속하게 프로젝트를 생성할 수 있는 커맨드라인 유틸리티입니다. Python 패키지 프로젝트 생성 등에 이상적입니다.

### 주요 기능

- **크로스 플랫폼:** Windows, Mac, Linux 지원
- **사용자 친화적:** Python 지식 불필요
- **다양성:** Python 3.7부터 3.12까지 호환
- **다중 언어 지원:** 모든 언어 또는 마크업 형식의 템플릿 사용 가능

### 설치 방법

```bash
# pipx 사용 권장
pipx install cookiecutter

# pipx를 사용할 수 없는 경우
python -m pip install --user cookiecutter
```
