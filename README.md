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

## 📁 파일 구조

```
langchain-practice/
├── 01_basic_lcel.py              # LCEL 기본 체인 (invoke)
├── 02_lcel_streaming.py          # 스트리밍 출력
├── 03_lcel_parallel.py           # 병렬 체인 (RunnableParallel)
├── 04_lcel_langsmith.py          # LangSmith 추적
├── 05_final_rag_project.py       # 통합 RAG 챗봇 (대화형)
├── practice.py                   # ⭐ 최종 실습 파일 (유사도 기반 RAG)
├── pdf_embedding_faiss.py        # PDF 임베딩 (참고)
├── data/
│   └── travel_guide.txt          # 서울 여행 가이드 데이터
├── requirements.txt              # 패키지 의존성
├── .env                          # API 키 설정 (필수)
└── .gitignore                    # Git 제외 파일
```

---

## 🚀 설치 및 실행

### 1. 저장소 클론
```bash
git clone https://github.com/heejeeong/langchain-practice.git
cd langchain-practice
```

### 2. 가상환경 생성 및 활성화
```bash
python -m venv .venv3
source .venv3/bin/activate  # Windows: .venv3\Scripts\activate
```

### 3. 패키지 설치
```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정
`.env` 파일을 생성하고 다음 내용 추가:

```env
# OpenAI API Key (필수)
OPENAI_API_KEY=sk-your-api-key-here

# LangSmith (선택)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_PROJECT=langchain-lcel-practice
```

### 5. 실행

#### 📌 practice.py (메인 실습 파일)

**단일 질문 모드:**
```bash
python practice.py --query "남산타워 야경 어때?"
```

**대화 루프 모드:**
```bash
python practice.py
# 질문 입력 → 답변 → 반복 (exit로 종료)
```

**인덱스 재구축:**
```bash
python practice.py --build --fast
```

#### 📌 05_final_rag_project.py (통합 챗봇)
```bash
python 05_final_rag_project.py
```

---

## 💡 주요 기능 설명

### 1️⃣ RAG 검색 + 유사도 판단
```python
# practice.py 핵심 로직
results = vectorstore.similarity_search_with_score(question, k=3)
best_score = results[0][1]

if best_score <= 0.7:  # 임계값
    # RAG 모드: 문서 컨텍스트 포함
    prompt = create_rag_prompt_with_context(context_docs)
else:
    # Fallback LLM 모드: 일반 답변
    prompt = create_fallback_prompt()
```

### 2️⃣ LCEL 파이프라인
```python
chain = prompt | llm | StrOutputParser()

# 스트리밍 출력
for chunk in chain.stream({"question": question}):
    print(chunk, end="", flush=True)
```

### 3️⃣ 섹션 기반 메타데이터
```python
# travel_guide.txt를 문단별로 분리
documents = [
    Document(
        page_content=section,
        metadata={
            "section_num": i,
            "section_title": "경복궁: 조선 시대의..."
        }
    )
]
```

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

## 🔧 커스터마이징

### 데이터 변경
`data/travel_guide.txt`를 수정하고 재구축:
```bash
python practice.py --build
```

### 임계값 조정
`practice.py`의 `process_single_query()` 함수:
```python
def process_single_query(vectorstore, llm, question: str, 
                        threshold: float = 0.7,  # 여기를 조정
                        k: int = 3):
```

### 임베딩 모델 변경
```python
# practice.py 상단 설정
EMBEDDING_MODEL = "BAAI/bge-m3"  # 정확도 높음, 느림
EMBEDDING_MODEL_FAST = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 빠름

# 실행 시 --fast 옵션으로 경량 모델 사용
python practice.py --build --fast
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

---

## 🎓 제출 정보

- **과목**: 생성형 AI 서비스 개발의 이해
- **구현 내용**: LangChain + RAG 기반 여행 챗봇
- **핵심 기술**: LCEL, FAISS, OpenAI, HuggingFace, LangSmith
- **특징**: 유사도 기반 자동 분기 (RAG/LLM)

---

## 🐛 트러블슈팅

### Q1. FAISS 인덱스가 없다고 나옵니다
```bash
python practice.py --build
```

### Q2. OpenAI API 키 오류
`.env` 파일에 `OPENAI_API_KEY` 설정 확인

### Q3. 임베딩 모델 로딩이 느립니다
```bash
python practice.py --build --fast  # 경량 모델 사용
```

### Q4. 한글이 깨집니다
터미널 인코딩을 UTF-8로 설정

---

## 📞 문의

GitHub Repository: https://github.com/heejeeong/langchain-practice

---

## 📝 라이선스

본 프로젝트는 학습 목적으로 제작되었습니다.
