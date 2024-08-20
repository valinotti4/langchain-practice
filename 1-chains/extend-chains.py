from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda

model = ChatOllama(model="llama3.1")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

prepend_text = RunnableLambda(lambda x: "-----HOLA ESTE TEXTO ESTA ANTES-----\n" + x)
word_count = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")


# Combined chain LCEL
chain = prompt_template | model | StrOutputParser() | word_count | prepend_text

result = chain.invoke({"topic": "panda", "joke_count": 10})
print(result)
