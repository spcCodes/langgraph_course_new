# here we are invoking a llm in a langgraph and then using a function to convert the output to uppercase

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

def response_from_llm(input_text):
    '''
    this function return a response from an llm
    '''
    response = model.invoke(input_text)
    return response.content


def convert_to_uppercase(input_text):
    '''
    this function will convert the input text to uppercase
    '''
    return input_text.upper()


from langgraph.graph import Graph

workflow = Graph()
workflow.add_node("Get_ans_from_llm" , response_from_llm)
workflow.add_node("Convert_to_uppercase" , convert_to_uppercase)
workflow.add_edge("Get_ans_from_llm" , "Convert_to_uppercase")
workflow.set_entry_point("Get_ans_from_llm")
workflow.set_finish_point("Convert_to_uppercase")
app = workflow.compile()

if __name__ == "__main__":
    result = app.invoke("What is the capital of France?")
    print(result)