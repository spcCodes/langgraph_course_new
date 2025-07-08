#creating a  simple langgraph graph

def greet_user(name):
    '''
    This function will greet the user with a message
    '''
    return f"Hello {name}!"


def convert_to_uppercase(input_word):
    '''
    This function will convert the input word to uppercase
    '''
    return input_word.upper()


from langgraph.graph import Graph

workflow = Graph()
workflow.add_node("User_greetings" , greet_user)
workflow.add_node("Uppercase_converter" , convert_to_uppercase)
workflow.add_edge("User_greetings" , "Uppercase_converter")
workflow.set_entry_point("User_greetings")
workflow.set_finish_point("Uppercase_converter")
app = workflow.compile()


if __name__ == "__main__":
    result = app.invoke("Suman")
    print(result)