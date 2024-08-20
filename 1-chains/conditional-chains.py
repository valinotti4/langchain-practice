from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableBranch

model = ChatOllama(model="llama3.1")


positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Generate a thank you note for this positive feedback: {feedback}."),
    ]
)
negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Generate a response addressing this negative feedback: {feedback}."),
    ]
)
neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        (
            "human",
            "Generate a request for more details about this neutral feedback: {feedback}.",
        ),
    ]
)
escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        (
            "human",
            "Generate a a message to escalate this feedback to a human agent: {feedback}.",
        ),
    ]
)

classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        (
            "human",
            "Classify the sentiment of this feedback as positive, negative, neutral or escalate: {feedback}.",
        ),
    ]
)

branches = RunnableBranch(
    (lambda x: "positive" in x, positive_feedback_template | model | StrOutputParser()),
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser(),
    ),
    (lambda x: "neutral" in x, neutral_feedback_template | model | StrOutputParser()),
    escalate_feedback_template | model | StrOutputParser(),
)

classification_chain = classification_template | model | StrOutputParser()

# Combined chain LCEL
chain = classification_chain | branches

# Feedback examples
# Good: This refurbished Alexa is perfect! Not a scratch on it! Great sound. I’m glad I purchased it.
# Negative: I received my order with the back area cracked and missing the charging cable. The delivery was so poor, I was extremely upset and ended up throwing the Echo in the garbage. This has been a frustrating and upsetting experience. I expected much better, and this has been a complete letdown. Would not recommend to buy the echo
# Neutral: The product is OK. Work as expected but nothing exceptional


result = chain.invoke(
    "It is ok but not much room. Got my older soon a different one which is 100x better and would trust if it was to accidentally fall. This I don't think would survive! Will do if got nothing else but there is better."
)
print(result)
