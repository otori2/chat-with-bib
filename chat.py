import os
import re
import sys
from pathlib import Path

import chainlit as cl
from chainlit.context import context
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from openai import AsyncOpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY", default="sk-")
index_dir = os.getenv("FAISS_DIR", default="faiss_index")
model_embedding = os.getenv("EMBEDDING_MODEL", default="text-embedding-ada-002")
model_chat = os.getenv("CHAT_MODEL", default="gpt-3.5-turbo")

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
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "query"])

client = AsyncOpenAI(
    api_key=api_key,
)


def revise_reference_format(text):
    pattern = re.compile(r"(\[\d+\])+")
    matches = pattern.findall(text)
    for match in matches:
        text = text.replace(f"。{match}", f"{match}。")
        text = text.replace(f"，{match}", f"{match}，")
        text = text.replace(f"．{match}", f"{match}．")
    return text


@cl.on_chat_start
def start_chat():
    cl.user_session.set(
        # Reference: https://github.com/Azure-Samples/azure-search-openai-demo/blob/main/app/backend/approaches/retrievethenread.py
        "message_history",
        [
            {
                "role": "system",
                "content": """\
You are an intelligent assistant. Use 'you' to refer to the individual asking the questions even if they ask with 'I'. Answer the following question using only the data provided in the sources below. For tabular information return it as an html table. Do not return markdown format. Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. If you cannot answer using the sources below, say you don't know. Use below example to answer. Please be sure to indicate the source of your information when replying.""",
            },
            {
                "role": "user",
                "content": """\
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
            },
            {
                "role": "assistant",
                "content": """\
In-network deductibles are $500 for employee and $1000 for family [info1.txt] and Overlake is in-network for the employee plan [info2.pdf][info4.pdf].""",
            },
        ],
    )


async def stream_token(msg, token: str, ref_numbers: dict[str, int], is_sequence=False):
    # Override MessageBase.stream_token
    token = token.replace("、", "，")
    token = token.replace("。", "．")

    if is_sequence:
        msg.content = token
    else:
        msg.content += token

    for k in ref_numbers.keys():
        target = f"[{k}]"
        if target in msg.content:
            if ref_numbers[k] == 0:
                new_value = max(ref_numbers.values()) + 1
                ref_numbers[k] = new_value
            token = revise_reference_format(
                token.replace(target, f"[{str(ref_numbers[k])}]")
            )
            msg.content = revise_reference_format(
                msg.content.replace(target, f"[{str(ref_numbers[k])}]")
            )

    await context.emitter.send_token(id=msg.id, token=token, is_sequence=is_sequence)


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
    content = prompt.format(
        context="\n\n".join(
            [
                f'[{doc.metadata["ID"]}-{doc.metadata["page"]}]:\n{doc.page_content}'
                for doc in docs
            ]
        ),
        query=message.content,
    )

    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": content})

    msg = cl.Message(content="")
    await msg.send()

    stream = await client.chat.completions.create(
        messages=message_history,
        stream=True,
        model=model_chat,
        temperature=0.0,
    )

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await stream_token(msg, token, ref_numbers)
            # await msg.stream_token(token)

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

    message_history.append({"role": "assistant", "content": msg.content})
    await msg.update()
