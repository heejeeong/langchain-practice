# 🌍 LangChain RAG 여행 챗봇 실습 프로젝트

LangChain + LCEL + RAG + LangSmith를 활용한 서울 여행 가이드 챗봇 구현

---

## 📋 프로젝트 개요

이 프로젝트는 **LangChain**을 기반으로 RAG(Retrieval-Augmented Generation) 시스템을 구현한 여행 정보 챗봇입니다. 사용자의 질문을 FAISS 벡터 데이터베이스에서 검색하고, 유사도에 따라 RAG 답변 또는 일반 LLM 답변을 생성합니다.

### 핵심 기능
- ✅ **FAISS 벡터 검색**: HuggingFace 임베딩으로 의미 기반 문서 검색
- ✅ **유사도 기반 분기**: 임계값에 따라 RAG/LLM 자동 선택
- ✅ **스트리밍 답변**: 토큰 단위 실시간 출력
- ✅ **LangSmith 추적**: 전체 체인 실행 과정 모니터링
- ✅ **LCEL 파이프라인**: Pipe(`|`) 연산자로 체인 구성

---

## 🛠️ 기술 스택

| 구성 요소 | 기술 |
|---------|------|
| **Framework** | LangChain, LCEL |
| **LLM** | OpenAI GPT-4o-mini |
| **임베딩** | HuggingFace (BAAI/bge-m3, paraphrase-multilingual-MiniLM) |
| **벡터 DB** | FAISS |
| **모니터링** | LangSmith |
| **언어** | Python 3.11+ |

---

## 💡 주요 기능 설명

### 1️. RAG 검색 + 유사도 판단


### 2️. LCEL 파이프라인


### 3️. 섹션 기반 메타데이터

---

## 📊 실행 예시

### ✅ 성공 케이스 (RAG 모드)
```bash
$ python practice.py --query "광장시장에서 뭐 먹을 수 있어?"

🔍 RAG 검색 중...
📊 최고 유사도: 0.45 (임계값: 0.7)
  [1] 🟢 0.45 | 섹션 8: 광장시장: 전통 시장으로...
  [2] 🟢 0.62 | 섹션 15: 음식 문화: 서울은 한식...

✅ 자료에서 답변 생성 (RAG 모드)

🤖 답변 (스트리밍): 광장시장에서는 빈대떡과 육회 같은 한국 전통 
음식을 맛볼 수 있습니다. 또한 오징어, 떡, 튀김 등 다양한 먹거리
가 있어...
```

### ⚠️ Fallback 케이스 (일반 LLM)
```bash
$ python practice.py --query "부산의 맛집 추천해줘"

🔍 RAG 검색 중...
📊 최고 유사도: 1.25 (임계값: 0.7)
  [1] 🔴 1.25 | 섹션 5: 명동: 쇼핑과 길거리...
  [2] 🔴 1.38 | 섹션 15: 음식 문화: 서울은...

⚠️  자료에서 답을 찾을 수 없습니다. 일반 LLM 답변 생성

🤖 답변 (스트리밍): 죄송하지만 제공된 자료는 서울 여행 정보
입니다. 부산의 맛집 정보는 포함되어 있지 않습니다. 부산을 방문
하신다면...
```

---

## 📚 학습 내용

### LCEL (LangChain Expression Language)
- `|` 파이프 연산자로 체인 구성
- `invoke()`, `stream()`, `batch()` 메서드
- `RunnableParallel`로 병렬 실행

### RAG (Retrieval-Augmented Generation)
- 문서 청킹 (Chunking)
- 벡터 임베딩 (Embeddings)
- 유사도 검색 (Similarity Search)
- 컨텍스트 기반 답변 생성

### LangSmith
- 체인 실행 추적
- 입/출력 로깅
- 성능 모니터링
