import streamlit as st
import main
import asyncio
import redirect as rd

def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

def run(query):
    if query:
        with rd.stdout() as out:
            ox = main.agent_chain.run(query) 
        output = out.getvalue()
        output = main.remove_formatting(output)
        st.write(ox.response)
        return True
    
st.set_page_config(layout = "wide")

with st.sidebar:
    st.title("LegalEase: \nLegal Document Assistant")
    st.write("""
    Generative AI powered legal document creator or drafter. automatically generate a wide range of legal documents, including contracts, agreements, wills, deeds, and more. Users can input specific details, such as names, dates, and terms, and the AI system produces a customized document based on predefined templates and legal language.
    """)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Send a message"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = run(prompt)
    with st.chat_message("LegalEase"):
        with st.spinner("Generating..."):
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})