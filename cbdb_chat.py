import openai
import streamlit as st
import os
import time

password =  st.sidebar.text_input('Give me password', type='password')
assistant_id = st.secrets.ASSISTANT_ID
openai.api_key = st.secrets.OPENAI_API_KEY

st.title('💬LODox Concept Chat Demo')
st.caption("LODox Concept Chat powered by OpenAI LLM")

# Streamlit UI for sidebar configuration
st.sidebar.title("Configuration")

# Sidebar for selecting the assistant
assistant_option = st.sidebar.selectbox(
        "Select an Assistant",
        ("LLaMaC", "building......")
    )

client = openai

if "start_chat" not in st.session_state:
    st.session_state.start_chat = False
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if st.sidebar.button("Start Chat"):
    st.session_state.start_chat = True
    thread = client.beta.threads.create()
    st.session_state.thread_id = thread.id

if st.sidebar.button("Exit Chat"):
    st.session_state.messages = []  # Clear the chat history
    st.session_state.start_chat = False  # Reset the chat state
    st.session_state.thread_id = None

if st.session_state.start_chat:
    if "openai_model" not in st.session_state:
        st.session_state.openai_model = "gpt-4-turbo-preview"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if password != st.secrets.PASSWORD:
        st.info("Wrong password")
        st.stop()
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Start conversation"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        client.beta.threads.messages.create(
                thread_id=st.session_state.thread_id,
                role="user",
                content=prompt
            )
        
        run = client.beta.threads.runs.create(
            thread_id=st.session_state.thread_id,
            assistant_id=assistant_id,
        )

        while run.status != 'completed':
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(
                thread_id=st.session_state.thread_id,
                run_id=run.id
            )
        messages = client.beta.threads.messages.list(
            thread_id=st.session_state.thread_id
        )

        # Process and display assistant messages
        assistant_messages_for_run = [
            message for message in messages 
            if message.run_id == run.id and message.role == "assistant"
        ]
        for message in assistant_messages_for_run:
            st.session_state.messages.append({"role": "assistant", "content": message.content[0].text.value})
            with st.chat_message("assistant"):
                st.markdown(message.content[0].text.value)

else:
    if assistant_option == "LLaMaC": 
        st.markdown("""
                # Welcome to LLaMaC, your Cultural Heritage Knowledge Base Assistant!

        LLaMaC is designed to assist users in managing cultural heritage information using the CIDOC Conceptual Reference Model (CIDOC-CRM) as its foundation. This system allows users to add, update, delete, and query cultural heritage data in a structured and reliable manner. Follow the guidelines below to interact with LLaMaC effectively:

        ## Adding Information
        - To add new cultural heritage information, use the format `ADD[Your Information Here]`. LLaMaC will evaluate the authenticity of the information. If considered genuine, it will be added to the knowledge base.

        ## Updating Information
        - If you need to modify existing information, use `UPDATE[Your Updated Information Here]`. Ensure your updates are accurate and relevant to the cultural heritage data.

        ## Deleting Information
        - To remove information from the knowledge base, use `DEL[Specific Information to Delete]`. This action will permanently delete the specified data.

        ## Querying Information
        - Have a question? Use `Q[Your Question Here]` to receive answers based solely on the stored knowledge base data, without additional explanations or external information.

        ## Additional Instructions
        - For any adjustments or specific instructions on how data should be handled, use `CON[Your Instruction Here]`. LLaMaC will adapt its approach to meet these guidelines.

        ## Handling URLs
        - When provided with a URL in the format `ADD[URL]`, LLaMaC will retrieve, download, and extract cultural heritage data from the given link, incorporating this information into the knowledge base according to CIDOC-CRM standards.

        LLaMaC is committed to maintaining a high accuracy level and relies on users to provide genuine cultural heritage information. Your cooperation is essential in building a valuable and trustworthy knowledge base.

            """)
    if assistant_option == "building......": 
        st.markdown("""
                # Still building more 
                """)