"""
CBDB Chat Application

Environment Setup:
Configure your secrets in .streamlit/secrets.toml:
    OPENAI_API_KEY = "your_openai_api_key_here"
    ASSISTANT_ID = "your_assistant_id_here"
    PASSWORD = "CBDB"
    NEO4J_URI = "neo4j+s://your-instance.databases.neo4j.io"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "your_neo4j_password"
"""

import openai
import streamlit as st
import time
from neo4j import GraphDatabase
from cbdb_agents import CBDBAgenticRAG

password = st.sidebar.text_input('Give me password', type='password')
assistant_id = st.secrets.ASSISTANT_ID
openai.api_key = st.secrets.OPENAI_API_KEY

# Neo4j Configuration (Read-only mode)
NEO4J_URI = st.secrets.NEO4J_URI
NEO4J_USERNAME = st.secrets.NEO4J_USERNAME
NEO4J_PASSWORD = st.secrets.NEO4J_PASSWORD
NEO4J_AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)

# Initialize Neo4j read-only connection
@st.cache_resource
def init_neo4j_driver():
    """Initialize Neo4j driver with read-only access"""
    if NEO4J_URI:
        try:
            driver = GraphDatabase.driver(
                NEO4J_URI, 
                auth=NEO4J_AUTH,
                # Configure connection settings
                max_connection_lifetime=30 * 60,  # 30 minutes
                max_connection_pool_size=50,
                connection_acquisition_timeout=30  # 30 seconds
            )
            # Test connection
            driver.verify_connectivity()
            st.success("✅ Neo4j connection established (Read-only mode)")
            return driver
        except Exception as e:
            st.error(f"❌ Failed to connect to Neo4j: {str(e)}")
            return None
    else:
        st.warning("⚠️ Neo4j URI not configured")
        return None

# Initialize Neo4j driver
neo4j_driver = init_neo4j_driver()

# Initialize CBDB Multi-Agent RAG System
@st.cache_resource
def init_cbdb_rag_system():
    """Initialize the CBDB Multi-Agent RAG system"""
    if neo4j_driver and openai.api_key:
        try:
            # Create OpenAI client
            client = openai.OpenAI(api_key=openai.api_key)
            
            # Initialize the multi-agent system
            cbdb_rag = CBDBAgenticRAG(neo4j_driver, client)
            st.success("✅ CBDB Multi-Agent RAG System initialized")
            return cbdb_rag
        except Exception as e:
            st.error(f"❌ Failed to initialize CBDB RAG system: {str(e)}")
            return None
    else:
        st.warning("⚠️ Cannot initialize CBDB RAG system - missing Neo4j or OpenAI connection")
        return None

# Initialize CBDB RAG system
cbdb_rag_system = init_cbdb_rag_system()

def execute_read_query(query, parameters=None):
    """Execute a read-only query against Neo4j database"""
    if not neo4j_driver:
        return None
    
    try:
        with neo4j_driver.session(default_access_mode='READ') as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    except Exception as e:
        st.error(f"Query execution error: {str(e)}")
        return None

st.title('💬CBDB Chat')
st.caption("CBDB Chat powered by OpenAI LLM")

# Streamlit UI for sidebar configuration
st.sidebar.title("Configuration")

# Sidebar for selecting the assistant
assistant_option = st.sidebar.selectbox(
        "Select an Assistant",
        ("CBDB Chat", "building......")
    )

# CBDB System Status in Sidebar
if assistant_option == "CBDB Chat":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏛️ CBDB System Status")
    
    # Neo4j Status
    if neo4j_driver:
        st.sidebar.success("✅ Neo4j Connected")
        
        # Try to get database statistics
        try:
            stats_query = """
            MATCH (p:Person) 
            RETURN count(p) as person_count
            """
            result = execute_read_query(stats_query)
            if result:
                person_count = result[0].get('person_count', 'Unknown')
                st.sidebar.info(f"📊 Persons in DB: {person_count:,}")
        except:
            st.sidebar.info("📊 Database statistics loading...")
    else:
        st.sidebar.error("❌ Neo4j Disconnected")
    
    # RAG System Status
    if cbdb_rag_system:
        st.sidebar.success("✅ Multi-Agent RAG Active")
    else:
        st.sidebar.error("❌ RAG System Inactive")
    
    # OpenAI Status
    if openai.api_key:
        st.sidebar.success("✅ OpenAI Connected")
    else:
        st.sidebar.error("❌ OpenAI Disconnected")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Query Tips")
    st.sidebar.markdown("""
    - Use specific names for better results
    - Ask about relationships between people
    - Specify dynasties or time periods
    - Try both English and Chinese names
    """)
    
    st.sidebar.markdown("### 🔍 Sample Queries")
    sample_queries = [
        "Tell me about Confucius",
        "Who were Li Bai's friends?",
        "Tang dynasty emperors",
        "Who lived in the 10th century?",
        "Students of Zhu Xi"
    ]
    
    for query in sample_queries:
        if st.sidebar.button(f"💬 {query}", key=f"sample_{hash(query)}"):
            st.session_state.sample_query = query

if "start_chat" not in st.session_state:
    st.session_state.start_chat = False

if st.sidebar.button("Start Chat"):
    st.session_state.start_chat = True

if st.sidebar.button("Exit Chat"):
    st.session_state.messages = []  # Clear the chat history
    st.session_state.start_chat = False  # Reset the chat state

if st.session_state.start_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if password != st.secrets.PASSWORD:
        st.info("Wrong password")
        st.stop()
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle sample queries
    if hasattr(st.session_state, 'sample_query'):
        prompt = st.session_state.sample_query
        del st.session_state.sample_query
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
    elif prompt := st.chat_input("Ask about Chinese historical figures and their relationships..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
    else:
        prompt = None
    
    if prompt:
        # Use CBDB Multi-Agent RAG System
        if cbdb_rag_system:
            with st.chat_message("assistant"):
                with st.spinner("Searching CBDB database and analyzing relationships..."):
                    try:
                        # Process query through multi-agent system
                        response = cbdb_rag_system.process_query(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"Error processing query: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            # Show error if RAG system is not available
            with st.chat_message("assistant"):
                error_msg = "❌ CBDB Multi-Agent RAG System is not available. Please check your Neo4j and OpenAI connections."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

else:
    if assistant_option == "CBDB Chat": 
        st.markdown("""
                # Welcome to CBDB Chat - Your Chinese Biographical Database Assistant!

        CBDB Chat is powered by a sophisticated multi-agent RAG (Retrieval-Augmented Generation) system designed specifically for exploring the China Biographical Database (CBDB). Our system intelligently queries a comprehensive database of 80,000+ Chinese historical figures and their complex relationships.

        ## What You Can Ask About:

        ### 📚 **Biographical Information**
        - Basic information about historical figures: *"Tell me about Li Bai"*
        - Birth/death dates, dynasties, and official positions
        - Chinese and English names, places of origin

        ### 🔗 **Relationships & Connections**  
        - Family relationships: *"Who were Confucius's descendants?"*
        - Teacher-student relationships: *"Who studied under Zhu Xi?"*
        - Political and social connections: *"Who served under Emperor Kangxi?"*
        - Professional relationships and networks

        ### 🏛️ **Historical Context**
        - People from specific dynasties: *"Show me Tang dynasty poets"*
        - Historical time periods: *"Who lived during 800-900 CE?"*
        - Regional connections and geographical relationships

        ### 🔍 **Complex Queries**
        - Multi-generational family trees
        - Social networks and influence patterns
        - Career trajectories and office progressions
        - Contemporary figures and their interactions

        ## How It Works:

        Our **Multi-Agent System** includes:
        - **🎯 Smart Router**: Automatically selects the best retrieval method for your question
        - **🔍 Specialized Retrievers**: Optimized for different types of CBDB queries
        - **🤖 Text2Cypher**: Converts complex questions into database queries
        - **✅ Answer Critic**: Ensures comprehensive and accurate responses
        - **🎨 Response Enhancement**: Uses AI to provide clear, contextual explanations

        ## Example Queries:
        - *"Who was the teacher of Wang Yangming?"*
        - *"List Song dynasty officials who served in Hangzhou"*
        - *"What relationships did Su Shi have with other literati?"*
        - *"Who lived between 1000-1100 CE and held the position of Prime Minister?"*

        **Start chatting to explore China's rich biographical heritage!**
            """)
    if assistant_option == "building......": 
        st.markdown("""
                # Still building more 
                """)