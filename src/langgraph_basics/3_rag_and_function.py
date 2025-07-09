from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough,RunnableParallel
from rag_ingestion import retriever
from langchain_openai import ChatOpenAI
from langgraph.graph import Graph
from utils import stream_output

model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)


def extract_topics(state):
    messages = state['messages']  ##asssuming state is a dict with messages key
    question = messages[-1]   ## Fetching the user question
    complete_query = "Your task is to extract topic from the following query . If the extracted topic is two words or more, the topic should be in the form of 'word1_word2' . If the extracted topic is one word, the topic should be in the form of 'word1' . Following is the user query: " + question
    response = model.invoke(complete_query)
    state['messages'].append(response.content)
    return state

def call_rag(state):
    messages = state['messages']
    question = messages[0] ## Fetching the user question

    template = """Answer the question based only on the following context:
    {context}
    Donot answer anything out of the context, if you dont find the answer in the context, then say "I don't know"
    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)

    retrieval_chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | prompt
        | model
        | StrOutputParser()
        )
    result = retrieval_chain.invoke(question)
    return result

workflow = Graph()
workflow.add_node("Extract_topics" , extract_topics)
workflow.add_node("Call_rag" , call_rag)
workflow.add_edge("Extract_topics" , "Call_rag")
workflow.set_entry_point("Extract_topics")
workflow.set_finish_point("Call_rag")
app = workflow.compile()


if __name__ == "__main__":
    input = {
        "messages": ["can you tell me about Bangladesh industrial growth?"]
    }

    response = app.invoke(input)
    print(stream_output(app, input))
    # print(response)



