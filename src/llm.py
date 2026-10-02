import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()


def get_llm():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set."
        )

    llm = ChatGroq(
        groq_api_key=api_key,
        model="openai/gpt-oss-20b",
        temperature=0.1,
        max_tokens=1024
    )

    return llm

