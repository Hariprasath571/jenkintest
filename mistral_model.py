from langchain_community.llms import Ollama
import streamlit as st

def generate_response(prompt):
    llm = Ollama(model="mistral")  # Update to the Mistral model name
    response = llm(prompt)
    print (response)
    return response

# Input question
question = st.text_input("Ask a question:")
print("Question: ", question)

if question:
    # Prepare the prompt for LLaMA 
    prompt = "Question: {question}"

    # Generate response using LLaMA
    answer = generate_response(prompt)
    print("ans is :", answer)

        # Display the generated response
    st.subheader("Generated Response:")
    st.write(answer)