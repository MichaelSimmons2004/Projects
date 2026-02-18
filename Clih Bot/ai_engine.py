def get_ai_response(message):
    # Mock: Later, replace with llm.invoke(message)
    return f"Echoing your message: {message} (AI placeholder)"

# Future: Add tool-calling, etc.
# from langchain_community.llms import Ollama
# llm = Ollama(model="your_model")
# def get_ai_response(message):
#     return llm(message)