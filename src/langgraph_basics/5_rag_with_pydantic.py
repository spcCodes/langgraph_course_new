from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph , START , END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage , AIMessage , SystemMessage
from typing import TypedDict , Annotated
from pydantic import BaseModel , Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough,RunnableParallel
from langgraph.graph import StateGraph , START , END
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


# Load the database and retriever from the "bangladesh_economy" persist directory
embeddings = OpenAIEmbeddings()
db = Chroma(persist_directory="bangladesh_economy", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

#create a state class
class GraphState(TypedDict):
    topic : Annotated[str, None]
    llm_response : Annotated[str, None]
    rag_response : Annotated[str, None]
    messages : Annotated[list, add_messages]


#creating pydantic classes for each module
class TopicSelector(BaseModel):
    """
    you have to extract the topic from the user query.
    """
    topic : str = Field(description="The topic of the user query")

topic_extracction_llm = model.with_structured_output(TopicSelector)

class RAGResponse(BaseModel):
    """
    You have to provide response from the RAG agent.
    """
    rag_response : str = Field(description="The response from the RAG agent")

rag_structured_llm = model.with_structured_output(RAGResponse)


class LLMResponse(BaseModel):
    """
    You have to call the LLM agent.
    """
    llm_response : str = Field(description="The response from the LLM agent")

structured_llm = model.with_structured_output(LLMResponse)


topic_parser = PydanticOutputParser(pydantic_object=TopicSelector)
rag_parser = PydanticOutputParser(pydantic_object=RAGResponse)
llm_parser = PydanticOutputParser(pydantic_object=LLMResponse)



#defining the nodes
def extract_topic(state):
    print('-> Calling topic extraction agent ->')

    topic_extraction_system_prompt = """
    You are a topic extraction agent. You will be given a user query and you will need to extract the topic of the user query.
    the topic may be related to Bangladesh or non Bangladesh related.
    

    Respond in json format with the following keys:
    
    topic: The topic of the user query.if the topic is related to Bangladesh, then output 'Bangladesh' else output 'Not Related'.

    """

    human_message = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    ai_message = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
    system_message = [SystemMessage(content=topic_extraction_system_prompt)]

    messages = system_message + ai_message + human_message
    
    
    topic_response = topic_extracction_llm.invoke(messages)

    # We only update the topic, not messages. The router will use the topic.
    return {
        "topic": topic_response.topic
    }


def call_rag(state):
    human_message = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    question = human_message[-1].content ## Fetching the last user question

    template = """Answer the question based only on the following context:
    {context}
    Donot answer anything out of the context, if you dont find the answer in the context or anything related to the context, then say "I don't know"
    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)

    retrieval_chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | prompt
        | rag_structured_llm
        )
    result = retrieval_chain.invoke(question)
    return {
        "messages": [AIMessage(content=result.rag_response)],
        "rag_response": result.rag_response
    }
    

def call_llm(state):
    print('-> Calling LLM agent ->')
    
    human_message = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    question = human_message[-1].content # Fetch the last human message

    calling_llm_system_prompt = """
    You are a helpful assistant that can answer questions about the user query.
    follwoing is the user question 
    {question}
    """

    ai_message = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
    system_message = [SystemMessage(content=calling_llm_system_prompt.format(question=question))]
    
    # We only need the history of messages for the LLM call, not the question again.
    messages = system_message + ai_message + human_message 

    llm_structured_response = structured_llm.invoke(messages)

    return {
        "messages": [AIMessage(content=llm_structured_response.llm_response)],
        "llm_response": llm_structured_response.llm_response
    }
    
def router(state):
    print('-> Calling Router agent ->')
    topic = state.get('topic')
    if topic == 'Bangladesh':
        return 'RAG_call'
    else:
        return 'LLM_call'

#defining the workflow

workflow = StateGraph(GraphState)
workflow.add_node("topic_extraction" , extract_topic)
workflow.add_node("RAG" , call_rag)
workflow.add_node("LLM" , call_llm)
workflow.add_edge(START , "topic_extraction")
workflow.add_conditional_edges("topic_extraction" ,
                               router,
                               {
                                   "RAG_call" : "RAG",
                                   "LLM_call" : "LLM"
                               }
                               )
workflow.add_edge("RAG" , END)
workflow.add_edge("LLM" , END)
app = workflow.compile()

if __name__ == "__main__":
    while True:
        user_input = input("Enter your question: ")
        if user_input.lower() == "exit":
            break
        result = app.invoke({"messages": [HumanMessage(content=user_input)]})
        print(result)
