##Run using command: "streamlit run /home/user/projects/SAP_GenAI/Advanced/orchestration-streamlit.py" 

## 0. AI Core Credential 
from config import init_access

## 1. Orchestration
## 1.1 Define template
from gen_ai_hub.orchestration.models.message import SystemMessage, UserMessage, Message
from gen_ai_hub.orchestration.models.template import Template, TemplateValue
template = Template(
    messages=[
            SystemMessage("You are a helpful chatbot assistant."),
            UserMessage("{{?user_query}}"),
    ]
)
## 1.2 Define LLM
from gen_ai_hub.orchestration.models.llm import LLM
llm = LLM(
    name="gpt-4o", 
    version="latest", 
    parameters={
        "max_tokens": 2560, 
        "temperature": 0.2
    }
)
## 1.3 Define Orchestration
from gen_ai_hub.orchestration.models.config import OrchestrationConfig
config = OrchestrationConfig(
    template=template,  # or use a referenced prompt template from Step 1
    llm=llm,
)
from gen_ai_hub.orchestration.service import OrchestrationService
orchestration_service = OrchestrationService(
   # api_url = deployment_url,
    config = config,
) 



## 2. Chat with history function using Orchestration
def chat(user_input, chat_history):
    response = orchestration_service.run(
        template_values=[
            TemplateValue(
                name="user_query", 
                value=user_input
            )        
        ],
        history=chat_history,
    )
    
    message = response.orchestration_result.choices[0].message
    
    if isinstance(response.module_results.templating, list):
        chat_history = response.module_results.templating
    else:
        chat_history = []

    chat_history.append(message)
    
    return message.content, chat_history

## 3. Streamlit UI

import streamlit as st

st.title("💬 Chat Assistant")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_content, st.session_state.chat_history = chat(
                prompt, 
                st.session_state.chat_history
            )
            st.markdown(response_content)
    
    # Add assistant message to chat
    st.session_state.messages.append({"role": "assistant", "content": response_content})
