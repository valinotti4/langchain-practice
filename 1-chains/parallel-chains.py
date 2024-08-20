from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel

model = ChatOllama(model="llama3.1")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)


def analyze_best(jokes):
    analyze_best_joke_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a comedy critic."),
            ("human", "Given these jokes: {jokes}. Select the best one."),
        ]
    )
    return analyze_best_joke_template.format_prompt(jokes=jokes)


def analyze_worst(jokes):
    analyze_worst_joke_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a comedy critic."),
            ("human", "Given these jokes: {jokes}. Select the worst one."),
        ]
    )
    return analyze_worst_joke_template.format_prompt(jokes=jokes)


best_joke_chain = RunnableLambda(lambda x: analyze_best(x)) | model | StrOutputParser()
worst_joke_chain = (
    RunnableLambda(lambda x: analyze_worst(x)) | model | StrOutputParser()
)


def combine_best_worst(best, worst):
    return f"BEST JOKE SELECTION:\n{best}\n\nWORST JOKE SELECTION:\n{worst}"


# Combined chain LCEL
chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"best": best_joke_chain, "worst": worst_joke_chain})
    | RunnableLambda(
        lambda x: combine_best_worst(x["branches"]["best"], x["branches"]["worst"])
    )
)

result = chain.invoke({"topic": "panda", "joke_count": 10})
print(result)
