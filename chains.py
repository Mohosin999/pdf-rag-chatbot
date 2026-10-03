from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.prompts import PromptTemplate

import config

SYSTEM_PROMPT = """You are a helpful assistant. Just answer the question from below context only:
1. If it's not relevant, say "This information is not in the document".
2. Don't assume.
3. Give information directly from the context.

Context: {context}

Question: {question}

Answer:"""


def build_qa_chain(llm, retriever):
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=config.MEMORY_K,
        output_key="answer",
    )

    qa_prompt = PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=["context", "question"],
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
    )
    return qa_chain
