"""
05. 최종 프로젝트: RAG + 병렬 분류 + 스트리밍 + LangSmith 추적
========================================================
여행 정보 챗봇 (Travel Guide Chatbot)

파이프라인:
1. 사용자 질문 입력
2. [병렬] 감정 분석 + 카테고리 분류
3. FAISS RAG 검색 (의미 기반)
4. 검색 결과에 따라 분기:
   - 검색 성공 → RAG 컨텍스트 + LLM 답변
   - 검색 실패 → Fallback LLM 답변
5. 스트리밍으로 토큰 단위 출력
6. LangSmith 자동 추적
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ============ 설정 ============
TRAVEL_GUIDE_PATH = Path(__file__).parent / "data" / "travel_guide.txt"
FAISS_INDEX_PATH = Path(__file__).parent / "faiss_index_travel"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEARCH_RESULTS_K = 3


def setup_langsmith():
    """LangSmith 트레이싱 설정"""
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
        print("[🔍 LangSmith] 트레이싱 활성화됨")
        print(f"[🔍 LangSmith] 프로젝트: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
    else:
        print("[ℹ️ LangSmith] 트레이싱 비활성화 (필요시 .env에서 설정)")


def load_and_chunk_guide(file_path: Path):
    """여행 가이드 문서 로드 및 청킹"""
    if not file_path.exists():
        raise FileNotFoundError(f"여행 가이드 파일 없음: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ":", " ", ""],
    )
    chunks = text_splitter.split_text(content)
    print(f"[📄 문서] {len(chunks)}개 청크 생성")
    return chunks


def create_embeddings():
    """HuggingFace 임베딩 모델"""
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    except:
        device = "cpu"
    
    print(f"[🔗 임베딩] 모델: {EMBEDDING_MODEL}, device: {device}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


def build_or_load_faiss(embeddings):
    """FAISS 인덱스 구축 또는 로드"""
    # 기존 인덱스 있으면 로드
    if (FAISS_INDEX_PATH / "index.faiss").exists():
        print(f"[📦 벡터DB] 기존 인덱스 로드: {FAISS_INDEX_PATH}")
        return FAISS.load_local(
            str(FAISS_INDEX_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    
    # 없으면 새로 구축
    print(f"[📦 벡터DB] 새 인덱스 구축")
    chunks = load_and_chunk_guide(TRAVEL_GUIDE_PATH)
    
    vectorstore = FAISS.from_texts(chunks, embeddings)
    FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_INDEX_PATH))
    print(f"[💾 저장] FAISS 인덱스 저장 완료")
    return vectorstore


def create_parallel_classifier():
    """감정 분석 + 카테고리 분류 체인 (병렬)"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 감정 분석
    sentiment_prompt = ChatPromptTemplate.from_messages([
        ("human", "다음 질문의 감정을 한 단어로만 답하세요 (흥분/평상심/피곤/불만): {question}")
    ])
    sentiment_chain = sentiment_prompt | llm | StrOutputParser()
    
    # 카테고리 분류
    category_prompt = ChatPromptTemplate.from_messages([
        ("human", "다음 질문을 분류하세요. 다음 중 하나로만: 관광지/음식/교통/숙박/기타\n질문: {question}")
    ])
    category_chain = category_prompt | llm | StrOutputParser()
    
    # 병렬 실행
    parallel_chain = RunnableParallel(
        sentiment=sentiment_chain,
        category=category_chain,
    )
    
    return parallel_chain


def search_faiss(vectorstore, query: str, k: int = SEARCH_RESULTS_K):
    """FAISS에서 유사 문서 검색"""
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results


def create_rag_prompt_with_context(context_docs: list):
    """문서 컨텍스트를 포함한 프롬프트 생성"""
    context_text = "\n\n".join([
        f"[주변 정보 {i+1}]\n{doc.page_content[:300]}"
        for i, doc in enumerate(context_docs)
    ])
    
    system_msg = f"""당신은 친절한 서울 여행 가이드 AI입니다.
아래의 여행 정보를 바탕으로 질문에 정확하고 간결하게 답변하세요.
답변할 때 정보의 출처를 자연스럽게 언급하세요.

=== 여행 정보 ===
{context_text}
================"""
    
    return ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "{question}"),
    ])


def create_fallback_prompt():
    """RAG 검색 실패 시 사용할 일반 프롬프트"""
    return ChatPromptTemplate.from_messages([
        ("system", "당신은 친절한 서울 여행 가이드 AI입니다. 여행에 관한 일반적인 조언을 해주세요."),
        ("human", "{question}"),
    ])


def main():
    print("=" * 60)
    print("🌍 최종 프로젝트: 여행 정보 RAG 챗봇")
    print("=" * 60)
    
    # LangSmith 설정
    setup_langsmith()
    
    # 임베딩 + FAISS 준비
    embeddings = create_embeddings()
    vectorstore = build_or_load_faiss(embeddings)
    
    # 병렬 분류기 생성
    parallel_classifier = create_parallel_classifier()
    
    # LLM 초기화
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    
    # ============ 대화 루프 ============
    print("\n💬 질문을 입력하세요 (exit 입력 시 종료):\n")
    
    while True:
        question = input("👤 질문: ").strip()
        
        if question.lower() == "exit":
            print("👋 종료합니다.")
            break
        
        if not question:
            continue
        
        print("\n" + "=" * 60)
        
        # [Step 1] 병렬 분류
        print("🔄 분석 중...")
        classification = parallel_classifier.invoke({"question": question})
        sentiment = classification["sentiment"]
        category = classification["category"]
        print(f"📊 감정: {sentiment} | 카테고리: {category}")
        
        # [Step 2] RAG 검색
        print("🔍 RAG 검색 중...")
        search_results = search_faiss(vectorstore, question, k=SEARCH_RESULTS_K)
        
        if search_results:
            # 검색 성공 - RAG 모드
            print(f"✅ {len(search_results)}개 관련 정보 발견 (신뢰도: {search_results[0][1]:.2f})")
            
            # 컨텍스트 추출
            context_docs = [doc for doc, score in search_results]
            prompt = create_rag_prompt_with_context(context_docs)
            chain = prompt | llm | StrOutputParser()
            
            # [Step 3] 스트리밍 답변
            print("\n🤖 답변 (스트리밍): ", end="")
            for chunk in chain.stream({"question": question}):
                print(chunk, end="", flush=True)
            print("\n")
        
        else:
            # 검색 실패 - Fallback LLM
            print("⚠️ 관련 정보 없음 → 일반 LLM 답변 모드")
            
            prompt = create_fallback_prompt()
            chain = prompt | llm | StrOutputParser()
            
            print("\n🤖 답변 (스트리밍): ", end="")
            for chunk in chain.stream({"question": question}):
                print(chunk, end="", flush=True)
            print("\n")
        
        print("=" * 60 + "\n")


if __name__ == "__main__":
    # 터미널 인코딩 설정
    try:
        if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
            sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    
    main()
