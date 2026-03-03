"""
01. 기본 LCEL 체인 실습
- PromptTemplate | ChatOpenAI | StrOutputParser
- invoke() 메서드로 동기 호출
"""

import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    # LCEL 체인: 프롬프트 → LLM → 파서
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 친절한 고객지원 AI입니다. 질문에 간결하고 정확하게 답변하세요."),
        ("human", "{question}"),
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = prompt | llm | StrOutputParser()

    # invoke: 단일 입력 동기 호출
    question = "환불 정책은 어떻게 되나요?"
    response = chain.invoke({"question": question})

    print("=" * 50)
    print("질문:", question)
    print("답변:", response)
    print("=" * 50)


if __name__ == "__main__":
    main()
