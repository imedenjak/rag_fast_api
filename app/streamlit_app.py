import streamlit as st
from dotenv import load_dotenv
from agent import build_graph

load_dotenv()

st.title("RAG Agent 🤖")


# Build agent once — cached so it doesn't rebuild on every interaction
@st.cache_resource
def get_agent():
    return build_graph()

agent = get_agent()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
if question := st.chat_input("Ask a question..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.invoke({"question": question})
            answer = result["answer"]
            st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
