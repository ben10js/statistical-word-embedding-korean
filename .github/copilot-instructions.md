# PoC 목적 및 Copilot 요청 예시

## 목적
“RAG + resonance scoring 기반 메타인지 에이전트 PoC 개발 중. 아주 작은 데이터셋, 빠른 의사코드 레벨의 파이프라인 구현 필요.”

## Copilot 요청 방식 예시
- 코드를 작성할 때 각 단계(데이터 로딩, 임베딩, 벡터DB 구축, 검색/스코어/출력)에 대해 주석, 설명, 예시 데이터를 꼭 포함해 주세요.
- 전체 실행 흐름과 원리가 초보자 입장에서 이해될 수 있게 설명해 주세요.
- 각 함수/클래스 정의 시 목적, 입력/출력, 쓰임새를 docstring으로 작성해 주세요.
- 최종적으로 내 입력에 대해 top-k 연관 기록 찾고, resonance score 계산하는 예시까지 샘플 데이터 따라 적어 주세요.

### 샘플 요청 구문

> Write a Python script that loads sample user notes, computes sentence embeddings, and stores them in a simple vector index — add comments explaining each step.

> 각 함수/클래스 정의할 때 목적, 입력/출력, 쓰임새를 docstring으로 써줘.

> 코드를 보고 처음 접하는 사람도 어떻게 돌아가는지 알 수 있게, 파이프라인 핵심 로직을 최대한 자세히 설명해줘.

> 최종적으로 내 입력에 대해 top-k 연관 기록 찾고, resonance score 계산하는 예시까지 샘플 데이터 따라 적어줘.
# Copilot Instructions for metacogai_via_hyperclova

## Project Overview
This project is a minimal Python application (see `app.py`) for experimenting with LSTM-based machine learning on document data. The workflow centers on ingesting JSON-formatted documents, preprocessing them, and running LSTM experiments using TensorFlow/Keras.

## Key Workflow
- **Data Location:** Source documents are stored as JSON files, typically in a user-specified directory (e.g., `C:\Users\User\OneDrive - konkuk.ac.kr\문서\n8n_metacog`).
- **Preprocessing:** Documents are loaded, tokenized, and converted to padded sequences for model input. Labels are extracted for supervised learning.
- **Modeling:** The main script builds and trains an LSTM model for text classification. Model parameters (embedding size, sequence length, etc.) are hardcoded but can be adjusted in `app.py`.
- **Output:** Results and processed data may be saved to a target directory (e.g., `C:\Users\User\Documents\n8n_json`).

## Patterns & Conventions
- All logic currently resides in `app.py`. There is no modularization or package structure.
- Data ingestion expects each JSON file to have at least `text` and `label` fields.
- TensorFlow/Keras is used for deep learning; ensure dependencies are installed before running.
- Paths are hardcoded for local Windows environment; update as needed for portability.

## Integration Points
- No external APIs or service integrations are present by default. If using Naver HyperCLOVA, add API calls and document the workflow.
- n8n is referenced as a data source/automation tool but not directly integrated in code.

## Developer Guidance
- To add new data sources, update the file reading logic in `app.py`.
- For new model architectures, modify the Keras model definition section.
- To automate workflows or integrate with n8n, consider adding API clients or file watchers.
- For large-scale experiments, refactor into modules and add configuration files.

## Example: Loading and Training
```python
json_folder = r"C:\Users\User\OneDrive - konkuk.ac.kr\문서\n8n_metacog"
# ... load JSON files, preprocess, train LSTM ...
```

## Key File
- `app.py`: Main script for all logic

---
_If any section is unclear or missing, please provide feedback for improvement._
