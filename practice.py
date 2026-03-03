import argparse
import sys
from pathlib import Path
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

load_dotenv()

# 설정
TXT_PATH = Path(__file__).parent / "data" / "travel_guide.txt"
FAISS_INDEX_PATH = Path(__file__).parent / "faiss_index_travel_guide"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
# BGE-M3: 정확도 높음, 느림 | paraphrase-multilingual-MiniLM: 빠름, 다국어
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_MODEL_FAST = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_txt(file_path: Path):
    """텍스트 파일 로드 - 섹션(문단)별로 분리"""
    if not file_path.exists():
        raise FileNotFoundError(
            f"텍스트 파일을 찾을 수 없습니다: {file_path}\n"
            f"data/ 폴더를 만들고 travel_guide.txt를 배치하세요."
        )

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 문단(\n\n)으로 분리해서 각각을 별도 문서로 만들기
    sections = [s.strip() for s in content.split("\n\n") if s.strip()]
    documents = []
    
    for i, section in enumerate(sections, 1):
        # 첫 번째 라인을 섹션 제목으로 사용 (예: "경복궁: ...")
        first_line = section.split("\n")[0]
        section_title = first_line[:40] if len(first_line) > 40 else first_line
        
        doc = Document(
            page_content=section,
            metadata={
                "source": str(file_path),
                "section_num": i,
                "section_title": section_title
            }
        )
        documents.append(doc)
    
    print(f"[로드] {len(documents)}개 섹션으로 분리 완료")
    return documents


def chunk_documents(documents):
    """문서 청킹"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"[청킹] {len(chunks)}개 청크 생성 (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def _get_device():
    """GPU/CPU 자동 감지 (cuda > mps > cpu)"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"  # Mac M1/M2
    except ImportError:
        pass
    return "cpu"


def create_embeddings(use_fast_model: bool = False):
    """HuggingFace 임베딩 모델 생성 (속도 최적화)"""
    device = _get_device()
    model = EMBEDDING_MODEL_FAST if use_fast_model else EMBEDDING_MODEL
    print(f"[임베딩] device={device}, batch_size=64")
    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 64,  # 배치 처리로 속도 향상
        },
    )


def build_faiss_index(chunks, embeddings):
    """FAISS 벡터 인덱스 구축"""
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("[FAISS] 벡터 인덱스 구축 완료")
    return vectorstore


def save_faiss_index(vectorstore):
    """FAISS 인덱스 저장"""
    FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_INDEX_PATH))
    print(f"[저장] FAISS 인덱스 저장: {FAISS_INDEX_PATH}")


def load_faiss_index(embeddings):
    """저장된 FAISS 인덱스 로드"""
    return FAISS.load_local(
        str(FAISS_INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def run_build(txt_path: Path, use_fast_model: bool = False):
    """txt 로드 → 청킹 → 임베딩 → FAISS 구축"""
    print("=" * 50)
    print("문서 로드 및 임베딩 처리 실습")
    print("=" * 50)

    documents = load_txt(txt_path)
    chunks = chunk_documents(documents)

    model = EMBEDDING_MODEL_FAST if use_fast_model else EMBEDDING_MODEL
    print(f"[임베딩] 모델 로딩: {model}")
    embeddings = create_embeddings(use_fast_model=use_fast_model)

    vectorstore = build_faiss_index(chunks, embeddings)
    save_faiss_index(vectorstore)

    return vectorstore


def run_query(vectorstore, query: str, k: int = 3, threshold: float = 0.7):
    """유사도 검색 실행 (유사도 점수 포함)"""
    results = vectorstore.similarity_search_with_score(query, k=k)
    print(f"\n질문: {query}\n")
    
    if not results:
        print("⚠️  검색 결과가 없습니다.\n")
        return
    
    # 가장 높은 유사도 확인
    best_score = results[0][1]
    if best_score > threshold:
        print(f"⚠️  유사도가 낮습니다 (최고: {best_score:.2f}). 자료에서 답을 찾을 수 없습니다.\n")
    
    for i, (doc, score) in enumerate(results, 1):
        section_num = doc.metadata.get('section_num', '?')
        section_title = doc.metadata.get('section_title', '정보')
        
        # 유사도 표시 (낮을수록 유사함 - L2 거리)
        similarity_status = "🟢" if score <= threshold else "🔴"
        print(f"[{i}] {similarity_status} 유사도: {score:.2f} | 섹션 {section_num}: {section_title}")
        print(doc.page_content[:250].replace("\n", " ") + "...")
        print()


def setup_langsmith():
    """LangSmith 트레이싱 설정 (환경변수 또는 코드)"""
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
        print("[🔍 LangSmith] 트레이싱 활성화됨")
        print(f"[🔍 LangSmith] 프로젝트: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
    else:
        print("[ℹ️  LangSmith] 트레이싱 비활성화")


def create_rag_prompt_with_context(context_docs: list):
    """문서 컨텍스트를 포함한 프롬프트 생성"""
    context_text = "\n\n".join([
        f"[참고 자료 {i+1}]\n{doc.page_content[:300]}"
        for i, doc in enumerate(context_docs)
    ])
    
    system_msg = f"""당신은 친절한 서울 여행 가이드 AI입니다.
아래의 여행 정보를 바탕으로 질문에 정확하고 간결하게 답변하세요.

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
        ("system", "당신은 친절한 여행 가이드 AI입니다. 제공된 자료에 답이 없으므로 일반적인 여행 조언을 해주세요."),
        ("human", "{question}"),
    ])


def main():
    print("=" * 60)
    print("🌍 서울 여행 RAG 챗봇 (유사도 기반 답변)")
    print("=" * 60)
    
    setup_langsmith()
    
    parser = argparse.ArgumentParser(description="텍스트 문서 로드 및 FAISS 벡터 DB 구축")
    parser.add_argument(
        "--txt",
        type=Path,
        default=TXT_PATH,
        help=f"텍스트 파일 경로 (기본: {TXT_PATH})",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="검색 쿼리 (인덱스가 있으면 검색만 수행)",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="강제로 인덱스 재구축",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="경량 모델 사용 (빠름, paraphrase-multilingual-MiniLM)",
    )
    args = parser.parse_args()

    embeddings = None
    vectorstore = None

    # 인덱스 로드 시도 (--build가 아니고 인덱스가 있는 경우)
    if not args.build and (FAISS_INDEX_PATH / "index.faiss").exists():
        print(f"[📦 벡터DB] 기존 FAISS 인덱스 로드: {FAISS_INDEX_PATH}")
        embeddings = create_embeddings(use_fast_model=args.fast)
        vectorstore = load_faiss_index(embeddings)

    # --build 또는 인덱스가 없으면 구축
    if vectorstore is None:
        vectorstore = run_build(args.txt, use_fast_model=args.fast)
    
    # LLM 초기화
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    
    # --query 옵션이 있으면 단일 질문 처리, 없으면 대화 루프
    if args.query:
        process_single_query(vectorstore, llm, args.query)
    else:
        # 대화 루프
        print("\n💬 질문을 입력하세요 (exit 입력 시 종료):\n")
        while True:
            question = input("👤 질문: ").strip()
            
            if question.lower() == "exit":
                print("👋 종료합니다.")
                break
            
            if not question:
                continue
            
            print("\n" + "=" * 60)
            process_single_query(vectorstore, llm, question)
            print("=" * 60 + "\n")


def process_single_query(vectorstore, llm, question: str, threshold: float = 0.7, k: int = 3):
    """단일 질문 처리: RAG 검색 → 유사도 판단 → 답변 생성"""
    
    # [Step 1] RAG 검색
    print("🔍 RAG 검색 중...")
    results = vectorstore.similarity_search_with_score(question, k=k)
    
    if not results:
        print("⚠️  검색 결과가 없습니다.")
        return
    
    # 가장 높은 유사도 확인
    best_score = results[0][1]
    print(f"📊 최고 유사도: {best_score:.2f} (임계값: {threshold})")
    
    # 검색된 문서 미리보기
    for i, (doc, score) in enumerate(results, 1):
        section_num = doc.metadata.get('section_num', '?')
        section_title = doc.metadata.get('section_title', '정보')
        similarity_status = "🟢" if score <= threshold else "🔴"
        print(f"  [{i}] {similarity_status} {score:.2f} | 섹션 {section_num}: {section_title}")
    
    # [Step 2] 유사도에 따라 분기
    if best_score <= threshold:
        # RAG 모드 (유사도 높음)
        print("\n✅ 자료에서 답변 생성 (RAG 모드)")
        context_docs = [doc for doc, score in results]
        prompt = create_rag_prompt_with_context(context_docs)
    else:
        # Fallback LLM 모드 (유사도 낮음)
        print("\n⚠️  자료에서 답을 찾을 수 없습니다. 일반 LLM 답변 생성")
        prompt = create_fallback_prompt()
    
    chain = prompt | llm | StrOutputParser()
    
    # [Step 3] 스트리밍 답변
    print("\n🤖 답변 (스트리밍): ", end="")
    for chunk in chain.stream({"question": question}):
        print(chunk, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    # 터미널 한글 출력 인코딩 (Windows 등)
    try:
        if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
            sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    main()
