"""
03. LCEL RunnableParallel 실습
- 여러 LLM 호출을 병렬로 동시 실행
"""

import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 감정 분석 체인
    sentiment_prompt = ChatPromptTemplate.from_messages([
        ("human", "다음 문장의 감정을 한 단어로만 답하세요 (긍정/부정/중립): {text}"),
    ])
    sentiment_chain = sentiment_prompt | llm | StrOutputParser()

    # 카테고리 분류 체인
    category_prompt = ChatPromptTemplate.from_messages([
        ("human", "다음 문장을 분류하세요. 다음 중 하나로만 답하세요: 환불, 배송, 결제, 기타\n문장: {text}"),
    ])
    category_chain = category_prompt | llm | StrOutputParser()

    # 병렬 실행: 두 체인을 동시에 실행
    parallel_chain = RunnableParallel(
        sentiment=sentiment_chain,
        category=category_chain,
    )

    customer_message = "주문한 지 2주가 넘었는데 아직 배송이 안 왔어요. 정말 답답합니다."

    result = parallel_chain.invoke({"text": customer_message})

    print("=" * 50)
    print("고객 메시지:", customer_message)
    print("감정 분석:", result["sentiment"])
    print("카테고리:", result["category"])
    print("=" * 50)


if __name__ == "__main__":
    main()
