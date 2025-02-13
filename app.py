import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the GPT-2 model and tokenizer
model_name = 'path_to_your_finetuned_model'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

st.title('Medical Query Response Generator')
st.write('This is an app that generate responses to medical queries.')

# User input for medical query
query = st.text_input('Enter your medical query:')

# User input for response length
response_length = st.slider('Select response length:', min_value=10, max_value=300, value=50)

# Generate response when user submits query
if st.button('Generate Response'):
    if query:
        # Encode input query and generate response
        inputs = tokenizer.encode(query, return_tensors='pt')
        outputs = model.generate(inputs, max_length=response_length, num_return_sequences=1)

        # Decode and display the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write('Response:')
        st.write(response)
    else:
        st.write('Please enter a query to generate a response.')

if __name__ == '__main__':
    st.run()
