import os
import re
import sys
from pathlib import Path
from typing import AsyncIterator

import chainlit as cl
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.vectorstores import FAISS

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY", default="sk-")
index_dir = os.getenv("FAISS_DIR", default="faiss_index")
model_embedding = os.getenv("EMBEDDING_MODEL", default="text-embedding-ada-002")
model_chat = os.getenv("CHAT_MODEL", default="gpt-3.5-turbo")

stopwords = ["参考文献"]

# Load FAISS index
if not Path(index_dir).exists():
    print("Index dir not found")
    sys.exit(1)

embeddings = OpenAIEmbeddings(
    api_key=api_key,
    model=model_embedding,
)
db = FAISS.load_local(index_dir, embeddings)

prompt_template = """\
Sources:
{context}

Question:
{query}
"""
prompt_txt = PromptTemplate(
    template=prompt_template, input_variables=["context", "query"]
)

model = ChatOpenAI(streaming=True, api_key=api_key)


def revise_reference_format(text):
    pattern = re.compile(r"(\[\d+\])+")
    matches = pattern.findall(text)
    for match in matches:
        text = text.replace(f"。{match}", f"{match}。")
        text = text.replace(f"，{match}", f"{match}，")
        text = text.replace(f"．{match}", f"{match}．")
    # text = text.replace("$$", "$")
    text = re.sub(r"\\tag\{\d+(\.\d+)?\}", "", text)
    return text


def ref_formatter(text: str, ref_numbers: dict[str, int]) -> str:
    text = text.replace("、", "，").replace("。", "．")
    for k in ref_numbers.keys():
        target = f"[{k}]"
        if target in text:
            if ref_numbers[k] == 0:
                new_value = max(ref_numbers.values()) + 1
                ref_numbers[k] = new_value
            text = revise_reference_format(
                text.replace(target, f"[{str(ref_numbers[k])}]")
            )

    return text


def postprocess(text: str) -> str:
    text = revise_reference_format(text)
    for stopword in stopwords:
        if stopword in text:
            text = text[: text.index(stopword)].strip()
    return text


async def format_reference(input: AsyncIterator[str]) -> AsyncIterator[str]:
    ref_numbers = cl.user_session.get("ref_numbers")
    buffer = ""
    async for chunk in input:
        buffer += chunk.content
        n_bra = sum(1 for c in buffer if c == "[")
        n_ket = sum(1 for c in buffer if c == "]")
        if n_bra == n_ket:
            yield ref_formatter(buffer, ref_numbers)
            buffer = ""
        else:
            cnks = buffer.split("[")
            cnks = cnks[:1] + ["[" + c for c in cnks[1:]]
            txt = "".join(cnks[:-1])
            if txt != "":
                yield ref_formatter(txt, ref_numbers)
            buffer = cnks[-1]

    yield buffer


@cl.on_chat_start
def start_chat():
    cl.user_session.set(
        # Reference: https://github.com/Azure-Samples/azure-search-openai-demo/blob/main/app/backend/approaches/retrievethenread.py
        "message_history",
        [
            (
                "system",
                """\
You are an intelligent assistant. Use 'you' to refer to the individual asking the questions even if they ask with 'I'. Answer the following question using only the data provided in the sources below. For tabular information return it as an html table. Do not return markdown format. Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. If you cannot answer using the sources below, say you don't know. Use below example to answer. Please be sure to indicate the source of your information when replying.""",
            ),
            (
                "user",
                """\
Sources:
[info1.txt]:
deductibles depend on whether you are in-network or out-of-network. In-network deductibles are $500 for employee and $1000 for family. Out-of-network deductibles are $1000 for employee and $2000 for family.

[info2.pdf]:
Overlake is in-network for the employee plan.

[info3.pdf]:
Overlake is the name of the area that includes a park and ride near Bellevue.

[info4.pdf]:
In-network institutions include Overlake, Swedish and others in the region

Question:
What is the deductible for the employee plan for a visit to Overlake in Bellevue?""",
            ),
            (
                "assistant",
                """\
In-network deductibles are $500 for employee and $1000 for family [info1.txt] and Overlake is in-network for the employee plan [info2.pdf][info4.pdf].""",
            ),
        ],
    )


@cl.on_message
async def main(message: cl.Message):
    docs = db.similarity_search(message.content)
    ref_docs = {
        f'{doc.metadata["ID"]}-{doc.metadata["page"]}': {
            "name": f'{doc.metadata["author"]}, {doc.metadata["title"].replace(r"{", "").replace(r"}", "")} ({doc.metadata["year"]}), pp. {doc.metadata["page"]}.',
            "url": doc.metadata["url"],
            "content": doc.page_content,
        }
        for doc in docs
    }
    ref_numbers = {key: 0 for key in ref_docs.keys()}
    cl.user_session.set("ref_numbers", ref_numbers)

    content = prompt_txt.format(
        context="\n\n".join(
            [
                f'[{doc.metadata["ID"]}-{doc.metadata["page"]}]:\n{doc.page_content}'
                for doc in docs
            ]
        ),
        query=message.content,
    )

    message_history = cl.user_session.get("message_history")
    message_history.append(("user", content))

    messages = [
        (m1, m2.replace("{", "{{").replace("}", "}}")) for m1, m2 in message_history
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    chain = prompt | model | format_reference

    msg = cl.Message(content="")
    await msg.send()

    async for chunk in chain.astream(message.content):
        await msg.stream_token(chunk)
    msg.content = postprocess(msg.content)

    ref_numbers = {k: v for k, v in ref_numbers.items() if v > 0}
    if len(ref_numbers) > 0:
        ref_numbers = dict(sorted(ref_numbers.items(), key=lambda x: x[1]))
        msg.content += "\n\nReferences:"
        text_elements = []
        for k, v in ref_numbers.items():
            source_name = ref_docs[k]["name"]
            text_elements.append(
                cl.Text(
                    content=ref_docs[k]["content"]
                    + f'\n\n[link]({ref_docs[k]["url"]})',
                    name=source_name,
                )
            )
            msg.content += f"\n[{v}]. {source_name}"

        msg.elements = text_elements

    message_history.append(("assistant", msg.content))
    await msg.update()
