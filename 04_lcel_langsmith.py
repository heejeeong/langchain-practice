"""
04. LCEL + LangSmith 통합 실습
- LangSmith 트레이싱 활성화
- invoke, stream, batch 모두 활용
- LangSmith 대시보드에서 실행 추적 확인
"""

import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


def setup_langsmith():
    """LangSmith 트레이싱 설정 (환경변수 또는 코드)"""
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
        print("[LangSmith] 트레이싱 활성화됨")
        print(f"[LangSmith] 프로젝트: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
    else:
        print("[LangSmith] 트레이싱 비활성화 (LANGCHAIN_TRACING_V2=true, LANGCHAIN_API_KEY 설정 필요)")


def main():
    setup_langsmith()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 스타트업 고객지원 AI입니다. 친절하고 전문적으로 답변하세요."),
        ("human", "{question}"),
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm | StrOutputParser()

    # 1. invoke - 단일 호출
    print("\n--- invoke ---")
    response = chain.invoke({"question": "무료 체험 기간은 얼마인가요?"})
    print("답변:", response)

    # 2. stream - 스트리밍
    print("\n--- stream ---")
    print("답변: ", end="")
    for chunk in chain.stream({"question": "결제 수단은 어떤 게 있나요?"}):
        print(chunk, end="", flush=True)
    print()

    # 3. batch - 여러 질문 일괄 처리
    print("\n--- batch ---")
    questions = [
        "환불 기한은?",
        "영업시간이 어떻게 되나요?",
    ]
    inputs = [{"question": q} for q in questions]
    responses = chain.batch(inputs)
    for q, r in zip(questions, responses):
        print(f"Q: {q}\nA: {r}\n")

    print("=" * 50)
    print("LangSmith 대시보드에서 트레이스를 확인하세요: https://smith.langchain.com")


if __name__ == "__main__":
    main()
