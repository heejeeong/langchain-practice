"""
02. LCEL 스트리밍 실습
- stream() 메서드로 토큰 단위 실시간 출력
"""

import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 고객지원 AI입니다. 2-3문장으로 간결히 답변하세요."),
        ("human", "{question}"),
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm | StrOutputParser()

    question = "배송은 며칠 걸리나요?"

    print("질문:", question)
    print("답변 (스트리밍): ", end="")

    for chunk in chain.stream({"question": question}):
        print(chunk, end="", flush=True)

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
