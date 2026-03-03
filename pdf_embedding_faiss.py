"""
문서 로드 및 임베딩 처리 실습
- PDF 로드 (PyMuPDFLoader - 한글/CJK 지원)
- 청킹 (RecursiveCharacterTextSplitter)
- 임베딩 (HuggingFace BAAI/bge-m3)
- 벡터 DB (FAISS)
"""

import argparse
import sys
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 설정
PDF_PATH = Path(__file__).parent / "data" / "sample1.pdf"
FAISS_INDEX_PATH = Path(__file__).parent / "faiss_index_sample1"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
# BGE-M3: 정확도 높음, 느림 | paraphrase-multilingual-MiniLM: 빠름, 다국어
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_MODEL_FAST = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_pdf(file_path: Path):
    """PDF 파일 로드 (PyMuPDF - 한글/CJK 지원)"""
    if not file_path.exists():
        raise FileNotFoundError(
            f"PDF 파일을 찾을 수 없습니다: {file_path}\n"
            f"data/ 폴더를 만들고 sample1.pdf를 배치하세요."
        )

    loader = PyMuPDFLoader(str(file_path))
    documents = loader.load()
    print(f"[로드] {len(documents)}개 페이지 로드 완료")
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


def run_build(pdf_path: Path, use_fast_model: bool = False):
    """PDF 로드 → 청킹 → 임베딩 → FAISS 구축"""
    print("=" * 50)
    print("문서 로드 및 임베딩 처리 실습")
    print("=" * 50)

    documents = load_pdf(pdf_path)
    chunks = chunk_documents(documents)

    model = EMBEDDING_MODEL_FAST if use_fast_model else EMBEDDING_MODEL
    print(f"[임베딩] 모델 로딩: {model}")
    embeddings = create_embeddings(use_fast_model=use_fast_model)

    vectorstore = build_faiss_index(chunks, embeddings)
    save_faiss_index(vectorstore)

    return vectorstore


def run_query(vectorstore, query: str, k: int = 3):
    """유사도 검색 실행"""
    results = vectorstore.similarity_search(query, k=k)
    print(f"\n질문: {query}\n")
    for i, doc in enumerate(results, 1):
        print(f"[{i}] (페이지 {doc.metadata.get('page', '?')})")
        print(doc.page_content[:300].replace("\n", " ") + "...")
        print()


def main():
    parser = argparse.ArgumentParser(description="PDF 문서 로드 및 FAISS 벡터 DB 구축")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=PDF_PATH,
        help=f"PDF 파일 경로 (기본: {PDF_PATH})",
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
        print(f"[로드] 기존 FAISS 인덱스 로드: {FAISS_INDEX_PATH}")
        embeddings = create_embeddings(use_fast_model=args.fast)
        vectorstore = load_faiss_index(embeddings)

    # --build 또는 인덱스가 없으면 구축
    if vectorstore is None:
        vectorstore = run_build(args.pdf, use_fast_model=args.fast)

    # --query가 있으면 검색, 없으면 기본 테스트 쿼리
    query = args.query or "이 문서의 주요 내용은 무엇인가요?"
    print("\n--- 유사도 검색 ---")
    run_query(vectorstore, query)
    print("=" * 50)


if __name__ == "__main__":
    # 터미널 한글 출력 인코딩 (Windows 등)
    try:
        if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
            sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    main()
