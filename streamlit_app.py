import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import time
import io
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Enhanced Snowflake AI Assistant",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ff6b6b;
    margin: 0.5rem 0;
}
.success-metric {
    border-left-color: #28a745;
}
.warning-metric {
    border-left-color: #ffc107;
}
.info-metric {
    border-left-color: #17a2b8;
}
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    border-left: 4px solid #007bff;
    background-color: #f8f9fa;
}
.ai-response {
    border-left-color: #28a745;
    background-color: #f8fff9;
}
.user-message {
    border-left-color: #6f42c1;
    background-color: #f8f7ff;
}
.status-connected {
    color: #28a745;
    font-weight: bold;
}
.status-disconnected {
    color: #dc3545;
    font-weight: bold;
}
.limits-summary {
    background-color: #e8f4f8;
    padding: 0.75rem;
    border-radius: 0.5rem;
    border-left: 4px solid #17a2b8;
    font-size: 0.85rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


@dataclass
class QueryLimits:
    """Centralized query limits configuration"""
    default_preview: int = 100
    max_preview: int = 10000
    default_search: int = 1000
    max_search: int = 50000
    analytics_sample: int = 5000
    max_analytics: int = 100000
    table_list: int = 50
    knowledge_base: int = 10

    def validate(self):
        """Ensure limits are reasonable"""
        if self.default_preview > self.max_preview:
            self.default_preview = self.max_preview
        if self.default_search > self.max_search:
            self.default_search = self.max_search
        if self.analytics_sample > self.max_analytics:
            self.analytics_sample = self.max_analytics


# Import the MCP client
try:
    from snowflake_bridge import EnhancedSnowflakeServer  # Import the bridge
    USE_REAL_MCP = True
    print("✅ Successfully imported EnhancedSnowflakeServer from snowflake_bridge")
except ImportError as e:
    USE_REAL_MCP = False
    print(f"⚠️ Real MCP client not found: {e}")
    print("Using mock client for demo")


class MockMCPClient:
    """Mock MCP client for demonstration purposes"""

    def __init__(self):
        self.connected = False
        self.mock_tables = [
            {'TABLE_NAME': 'CUSTOMERS', 'TABLE_SCHEMA': 'PUBLIC', 'ROW_COUNT': 1000},
            {'TABLE_NAME': 'ORDERS', 'TABLE_SCHEMA': 'PUBLIC', 'ROW_COUNT': 5000},
            {'TABLE_NAME': 'PRODUCTS', 'TABLE_SCHEMA': 'SALES', 'ROW_COUNT': 500},
            {'TABLE_NAME': 'INVENTORY', 'TABLE_SCHEMA': 'WAREHOUSE', 'ROW_COUNT': 2000}
        ]

    def test_connection(self):
        """Mock connection test"""
        time.sleep(1)  # Simulate network delay
        self.connected = True
        return json.dumps({
            "status": "success",
            "message": "Connected to Snowflake successfully",
            "warehouse": "COMPUTE_WH",
            "database": "DEMO_DB",
            "schema": "PUBLIC"
        })

    def execute_query(self, query: str, timeout: int = 300):
        """Mock query execution"""
        time.sleep(0.5)  # Simulate query execution time

        query_upper = query.upper()

        if "SHOW TABLES" in query_upper:
            return json.dumps({
                "status": "success",
                "data": self.mock_tables,
                "row_count": len(self.mock_tables)
            })
        elif "SELECT" in query_upper:
            # Generate mock data based on query
            if "CUSTOMERS" in query_upper:
                mock_data = [
                    {"CUSTOMER_ID": i, "NAME": f"Customer {i}", "EMAIL": f"customer{i}@email.com", "CITY": "New York"}
                    for i in range(1, min(101, 1000))  # Limit to 100 rows for demo
                ]
            elif "ORDERS" in query_upper:
                mock_data = [
                    {"ORDER_ID": i, "CUSTOMER_ID": i % 100, "AMOUNT": 100.0 + i, "ORDER_DATE": "2024-01-01"}
                    for i in range(1, min(101, 5000))
                ]
            else:
                mock_data = [
                    {"ID": i, "VALUE": f"Sample {i}", "AMOUNT": 50.0 + i}
                    for i in range(1, 21)
                ]

            return json.dumps({
                "status": "success",
                "data": mock_data,
                "row_count": len(mock_data)
            })
        else:
            return json.dumps({
                "status": "success",
                "message": "Query executed successfully",
                "affected_rows": 1
            })

    # Add any other methods your real MCP client has
    def list_tables(self, database=None, schema=None, pattern=None, include_stats=False):
        """Mock list tables"""
        return json.dumps({
            "status": "success",
            "tables": self.mock_tables
        })

    def describe_table(self, table_name: str, include_sample: bool = False):
        """Mock describe table"""
        return json.dumps({
            "status": "success",
            "columns": [
                {"COLUMN_NAME": "ID", "DATA_TYPE": "NUMBER", "IS_NULLABLE": "NO"},
                {"COLUMN_NAME": "NAME", "DATA_TYPE": "VARCHAR", "IS_NULLABLE": "YES"},
                {"COLUMN_NAME": "EMAIL", "DATA_TYPE": "VARCHAR", "IS_NULLABLE": "YES"}
            ]
        })


class EnhancedSnowflakeApp:
    """Enhanced Snowflake application with improved architecture"""

    def __init__(self):
        # Initialize the MCP client
        if USE_REAL_MCP:
            try:
                self.mcp_client = EnhancedSnowflakeServer()
                print("✅ Successfully initialized Snowflake connection")
            except Exception as e:
                print(f"❌ Failed to initialize Snowflake server: {e}")
                print("Falling back to mock client...")
                self.mcp_client = MockMCPClient()
        else:
            print("Using mock MCP client for demo")
            self.mcp_client = MockMCPClient()

        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize all session state variables"""
        if 'query_limits' not in st.session_state:
            st.session_state.query_limits = QueryLimits()
            st.session_state.query_limits.validate()

        if 'query_history' not in st.session_state:
            st.session_state.query_history = []

        if 'current_data' not in st.session_state:
            st.session_state.current_data = None

        if 'connection_status' not in st.session_state:
            st.session_state.connection_status = 'disconnected'

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        if 'tables_cache' not in st.session_state:
            st.session_state.tables_cache = []

    def execute_query_with_enhanced_error_handling(self, query: str, timeout: int = 300):
        """Enhanced query execution with comprehensive error handling"""
        try:
            start_time = time.time()

            with st.spinner("Executing query..."):
                result = self.mcp_client.execute_query(query, timeout)

            execution_time = time.time() - start_time

            # Parse and validate result
            try:
                result_data = json.loads(result)
            except json.JSONDecodeError as e:
                st.error(f"❌ Failed to parse query result: {e}")
                return None

            # Store in query history with comprehensive metadata
            query_info = {
                'query': query,
                'status': 'success' if result_data.get('status') == 'success' else 'failed',
                'execution_time': execution_time,
                'row_count': len(result_data.get('data', [])) if result_data.get('data') else 0,
                'timestamp': datetime.now().isoformat(),
                'error': result_data.get('error') if result_data.get('status') != 'success' else None,
                'query_type': self.classify_query(query)
            }

            st.session_state.query_history.append(query_info)

            # Display results with enhanced formatting
            if result_data.get('status') == 'success':
                st.success(f"✅ Query executed successfully in {execution_time:.2f}s")

                if result_data.get('data'):
                    df = pd.DataFrame(result_data['data'])

                    # Show result summary
                    limits = st.session_state.query_limits
                    row_count = len(df)
                    st.info(f"📊 Retrieved {row_count:,} rows × {len(df.columns)} columns")

                    # Display data with pagination for large results
                    if row_count > limits.default_preview:
                        st.warning(f"⚠️ Large result set. Showing first {limits.default_preview:,} rows.")
                        st.dataframe(df.head(limits.default_preview), use_container_width=True)
                    else:
                        st.dataframe(df, use_container_width=True)

                    # Store current data for analysis
                    st.session_state.current_data = result_data['data']

                    # Quick analysis options
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("📊 Analyze Data", key=f"analyze_{int(time.time())}"):
                            self.render_data_statistics()
                    with col2:
                        if st.button("📈 Visualize", key=f"viz_{int(time.time())}"):
                            self.render_quick_visualizations(df)
                    with col3:
                        if st.button("💾 Export", key=f"export_{int(time.time())}"):
                            self.render_export_options(df)

                elif result_data.get('message'):
                    st.info(f"ℹ️ {result_data.get('message')}")
                else:
                    st.info("✅ Query executed successfully (no data returned)")
            else:
                error_msg = result_data.get('error', 'Unknown error occurred')
                st.error(f"❌ Query failed: {error_msg}")

        except Exception as e:
            execution_time = time.time() - start_time if 'start_time' in locals() else 0
            st.error(f"❌ Execution error: {str(e)}")

            # Store failed query in history
            st.session_state.query_history.append({
                'query': query,
                'status': 'error',
                'execution_time': execution_time,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'query_type': self.classify_query(query)
            })

    def classify_query(self, query: str) -> str:
        """Classify the type of query for analytics"""
        query_upper = query.upper()
        if any(word in query_upper for word in ['SELECT', 'SHOW', 'DESCRIBE']):
            return 'SELECT'
        elif any(word in query_upper for word in ['INSERT', 'UPDATE', 'DELETE']):
            return 'DML'
        elif any(word in query_upper for word in ['CREATE', 'ALTER', 'DROP']):
            return 'DDL'
        elif any(word in query_upper for word in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']):
            return 'ANALYTICAL'
        else:
            return 'OTHER'

    def render_enhanced_sidebar(self):
        """Enhanced sidebar with comprehensive controls"""
        st.sidebar.header("🔗 Connection Status")

        # Connection status with visual indicator
        status = st.session_state.get('connection_status', 'unknown')
        if status == 'connected':
            st.sidebar.markdown('<p class="status-connected">🟢 Connected</p>', unsafe_allow_html=True)
        else:
            st.sidebar.markdown('<p class="status-disconnected">🔴 Disconnected</p>', unsafe_allow_html=True)

        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Test Connection", key="test_conn"):
                self.test_connection()
        with col2:
            if st.button("Reset", key="reset_conn"):
                self.reset_connection()

        st.sidebar.divider()

        # Query Limits Configuration
        st.sidebar.header("⚙️ Query Limits")
        limits = st.session_state.query_limits

        with st.sidebar.expander("🔢 Configure Limits", expanded=False):
            limits.default_preview = st.number_input(
                "Default Preview Rows:",
                min_value=10,
                max_value=limits.max_preview,
                value=limits.default_preview,
                step=10,
                help="Number of rows to show by default"
            )

            limits.default_search = st.number_input(
                "Default Search Rows:",
                min_value=100,
                max_value=limits.max_search,
                value=limits.default_search,
                step=100,
                help="Maximum rows for search results"
            )

            limits.analytics_sample = st.number_input(
                "Analytics Sample Size:",
                min_value=1000,
                max_value=limits.max_analytics,
                value=limits.analytics_sample,
                step=1000,
                help="Sample size for statistical analysis"
            )

            # Quick presets
            st.write("**Quick Presets:**")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Fast", help="Smaller limits for speed"):
                    st.session_state.query_limits = QueryLimits(
                        default_preview=50,
                        default_search=500,
                        analytics_sample=1000,
                        table_list=20
                    )
                    st.rerun()
            with col2:
                if st.button("🔍 Detailed", help="Larger limits for analysis"):
                    st.session_state.query_limits = QueryLimits(
                        default_preview=1000,
                        default_search=10000,
                        analytics_sample=50000,
                        table_list=100
                    )
                    st.rerun()

        # Current limits summary
        st.sidebar.markdown(f"""
        <div class="limits-summary">
        <strong>📊 Current Limits:</strong><br>
        • Preview: <strong>{limits.default_preview:,}</strong> rows<br>
        • Search: <strong>{limits.default_search:,}</strong> rows<br>
        • Analytics: <strong>{limits.analytics_sample:,}</strong> rows<br>
        • Tables: <strong>{limits.table_list}</strong> shown
        </div>
        """, unsafe_allow_html=True)

        st.sidebar.divider()

        # Performance stats
        if st.session_state.query_history:
            st.sidebar.header("📊 Performance")

            total_queries = len(st.session_state.query_history)
            successful = len([q for q in st.session_state.query_history if q.get('status') == 'success'])
            success_rate = (successful / total_queries) * 100
            avg_time = np.mean([q.get('execution_time', 0) for q in st.session_state.query_history])

            st.sidebar.metric("Total Queries", total_queries)
            st.sidebar.metric("Success Rate", f"{success_rate:.1f}%")
            st.sidebar.metric("Avg Time", f"{avg_time:.2f}s")

            if st.sidebar.button("🗑️ Clear History"):
                st.session_state.query_history = []
                st.rerun()

    def render_dashboard(self):
        """Enhanced dashboard with metrics and insights"""
        st.header("📊 Enhanced Dashboard")

        # Welcome message
        st.markdown("""
        Welcome to your enhanced Snowflake AI Assistant! This dashboard provides an overview 
        of your query performance, connection status, and quick access to key features.
        """)

        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)

        query_history = st.session_state.get('query_history', [])

        with col1:
            st.metric("Total Queries", len(query_history))

        with col2:
            if query_history:
                successful = len([q for q in query_history if q.get('status') == 'success'])
                success_rate = (successful / len(query_history)) * 100
                st.metric("Success Rate", f"{success_rate:.1f}%", delta=f"{successful} successful")
            else:
                st.metric("Success Rate", "0%")

        with col3:
            if query_history:
                avg_time = np.mean([q.get('execution_time', 0) for q in query_history])
                st.metric("Avg Query Time", f"{avg_time:.2f}s")
            else:
                st.metric("Avg Query Time", "0.00s")

        with col4:
            current_data_rows = len(st.session_state.get('current_data', []))
            st.metric("Current Dataset", f"{current_data_rows:,} rows")

        # Query performance chart
        if len(query_history) > 1:
            st.subheader("📈 Query Performance Over Time")

            perf_data = []
            for i, query in enumerate(query_history):
                perf_data.append({
                    'Query Number': i + 1,
                    'Execution Time': query.get('execution_time', 0),
                    'Status': query.get('status', 'unknown'),
                    'Query Type': query.get('query_type', 'unknown')
                })

            perf_df = pd.DataFrame(perf_data)

            fig = px.line(
                perf_df,
                x='Query Number',
                y='Execution Time',
                color='Status',
                hover_data=['Query Type'],
                title="Query Execution Time Trend"
            )
            fig.update_layout(xaxis_title="Query Number", yaxis_title="Time (seconds)")
            st.plotly_chart(fig, use_container_width=True)

        # Recent activity
        st.subheader("📋 Recent Query Activity")
        if query_history:
            recent_queries = query_history[-5:]  # Last 5 queries

            activity_data = []
            for query in recent_queries:
                activity_data.append({
                    'Time': datetime.fromisoformat(query['timestamp']).strftime('%H:%M:%S'),
                    'Status': query['status'],
                    'Type': query.get('query_type', 'unknown'),
                    'Duration': f"{query.get('execution_time', 0):.2f}s",
                    'Rows': query.get('row_count', 0),
                    'Query': query['query'][:50] + '...' if len(query['query']) > 50 else query['query']
                })

            activity_df = pd.DataFrame(activity_data)
            st.dataframe(activity_df, use_container_width=True)
        else:
            st.info("No queries executed yet. Start exploring your data!")

        # Quick actions
        st.subheader("🚀 Quick Actions")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("📋 List Tables", key="quick_list_tables"):
                self.list_tables()

        with col2:
            if st.button("💬 AI Chat", key="quick_chat"):
                st.info("Navigate to the AI Chat tab to start a conversation!")

        with col3:
            if st.button("🔍 Execute Query", key="quick_query"):
                st.info("Navigate to the Query Tools tab to execute custom SQL!")

        with col4:
            if st.button("📊 Explore Data", key="quick_explore"):
                st.info("Navigate to the Data Explorer tab to browse your data!")

    def render_ai_chat(self):
        """Enhanced AI chat interface"""
        st.header("🤖 AI Chat Assistant")

        # AI status indicator
        st.info(
            "🔬 **Demo Mode**: This is a mock AI assistant for demonstration. In production, this would connect to actual AI services.")

        # Chat history display
        st.subheader("💬 Conversation History")
        if st.session_state.chat_history:
            for i, message in enumerate(st.session_state.chat_history):
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div class="chat-message user-message">
                    <strong>👤 You:</strong><br>
                    {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                elif message['role'] == 'assistant':
                    st.markdown(f"""
                    <div class="chat-message ai-response">
                    <strong>🤖 AI Assistant:</strong><br>
                    {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No conversation yet. Ask me anything about your Snowflake data!")

        # Chat input
        st.subheader("💬 Ask the AI")
        user_input = st.text_area(
            "What would you like to know about your data?",
            placeholder="Examples:\n• Show me the customers table\n• How many orders do we have?\n• What's in the products table?",
            height=100
        )

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if st.button("🤖 Send Message", key="send_ai_message"):
                if user_input:
                    self.process_ai_chat(user_input)

        with col2:
            if st.button("🗑️ Clear Chat", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()

        with col3:
            if st.button("💡 Get Suggestions", key="ai_suggestions"):
                self.show_ai_suggestions()

        # Quick suggestions
        st.subheader("💡 Quick AI Requests")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📋 Show me all tables", key="ai_show_tables"):
                self.process_ai_chat("Show me all tables")

        with col2:
            if st.button("👥 Explore customers", key="ai_customers"):
                self.process_ai_chat("Show me data from the customers table")

        with col3:
            if st.button("📊 Get summary stats", key="ai_stats"):
                self.process_ai_chat("Give me summary statistics for the current dataset")

    def process_ai_chat(self, user_input: str):
        """Process AI chat interaction with mock responses"""
        # Add user message
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })

        # Generate mock AI response based on input
        ai_response = self.generate_mock_ai_response(user_input)

        # Add AI response
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': ai_response,
            'timestamp': datetime.now().isoformat()
        })

        # Auto-execute suggested queries
        if "SHOW TABLES" in ai_response.upper():
            st.info("🔄 Executing suggested query...")
            self.execute_query_with_enhanced_error_handling("SHOW TABLES")
        elif "SELECT" in ai_response.upper():
            # Extract and execute SELECT query
            lines = ai_response.split('\n')
            for line in lines:
                if line.strip().upper().startswith('SELECT'):
                    st.info("🔄 Executing suggested query...")
                    self.execute_query_with_enhanced_error_handling(line.strip())
                    break

        st.rerun()

    def generate_mock_ai_response(self, user_input: str) -> str:
        """Generate mock AI responses based on user input"""
        user_lower = user_input.lower()

        if any(word in user_lower for word in ['table', 'tables', 'show']):
            return """I'll help you explore your tables. Let me show you what's available:

SHOW TABLES

This will display all the tables in your current database. You can then explore specific tables that interest you."""

        elif any(word in user_lower for word in ['customer', 'customers']):
            return """Great! Let me show you the customers data:

SELECT * FROM CUSTOMERS LIMIT 10

This will give you a preview of your customer data including customer IDs, names, emails, and other details."""

        elif any(word in user_lower for word in ['order', 'orders']):
            return """I'll retrieve your orders data for you:

SELECT * FROM ORDERS LIMIT 10

This will show you recent orders with order IDs, customer information, amounts, and dates."""

        elif any(word in user_lower for word in ['count', 'how many', 'number']):
            return """I can help you count records. Here are some useful counting queries:

For customers: SELECT COUNT(*) FROM CUSTOMERS
For orders: SELECT COUNT(*) FROM ORDERS

Would you like me to execute any of these counts?"""

        elif any(word in user_lower for word in ['stat', 'summary', 'analyze']):
            return """I can provide statistical analysis of your data. First, let me get some sample data and then I'll analyze it for you. What specific table would you like me to analyze?"""

        else:
            return f"""I understand you're asking about: "{user_input}"

I can help you with:
• Exploring tables and data
• Writing SQL queries
• Analyzing data patterns
• Getting statistics and summaries

Could you be more specific about what you'd like to do? For example:
- "Show me the customers table"
- "How many orders do we have?"
- "Analyze the sales data" """

    def show_ai_suggestions(self):
        """Show AI-powered suggestions"""
        suggestions = [
            "💡 Try asking: 'Show me all tables in the database'",
            "💡 Try asking: 'What's the structure of the customers table?'",
            "💡 Try asking: 'Give me a summary of recent orders'",
            "💡 Try asking: 'How many records are in each table?'",
            "💡 Try asking: 'Show me sample data from the products table'"
        ]

        for suggestion in suggestions:
            st.info(suggestion)

    def render_query_tools(self):
        """Enhanced query tools interface"""
        st.header("🔍 Enhanced Query Tools")

        limits = st.session_state.query_limits

        # Query input with better UX
        st.subheader("📝 SQL Query Editor")

        # Query templates
        with st.expander("📋 Query Templates", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📋 Show Tables"):
                    st.session_state.current_query = "SHOW TABLES"
                if st.button("👥 Customer Data"):
                    st.session_state.current_query = f"SELECT * FROM CUSTOMERS LIMIT {limits.default_preview}"
            with col2:
                if st.button("📦 Order Data"):
                    st.session_state.current_query = f"SELECT * FROM ORDERS LIMIT {limits.default_preview}"
                if st.button("📊 Count Records"):
                    st.session_state.current_query = "SELECT COUNT(*) FROM CUSTOMERS"

        # Main query input
        query = st.text_area(
            "Enter your SQL query:",
            value=st.session_state.get('current_query', ''),
            height=150,
            help="Write your SQL query here. Use the templates above for common queries."
        )

        # Query execution options
        col1, col2, col3 = st.columns(3)
        with col1:
            limit = st.number_input(
                "Row Limit",
                min_value=1,
                max_value=limits.max_search,
                value=limits.default_search,
                help="Maximum number of rows to return"
            )

        with col2:
            timeout = st.number_input(
                "Timeout (seconds)",
                min_value=1,
                max_value=600,
                value=300,
                help="Query timeout in seconds"
            )

        with col3:
            explain_query = st.checkbox(
                "Explain Query",
                help="Add EXPLAIN to see query execution plan"
            )

        # Execute query button
        if st.button("🚀 Execute Query", key="execute_main_query"):
            if query.strip():
                # Add LIMIT if not present and limit is specified
                final_query = query
                if limit and "LIMIT" not in query.upper() and "COUNT" not in query.upper():
                    final_query = f"{query.rstrip(';')} LIMIT {limit}"
                    st.info(f"Added LIMIT {limit:,} to your query for performance")

                if explain_query:
                    final_query = f"EXPLAIN {final_query}"

                self.execute_query_with_enhanced_error_handling(final_query, timeout)
            else:
                st.error("Please enter a query")

        # Query history
        if st.session_state.query_history:
            st.subheader("📜 Query History")

            history_data = []
            for i, query_info in enumerate(reversed(st.session_state.query_history[-10:])):  # Last 10
                history_data.append({
                    'Time': datetime.fromisoformat(query_info['timestamp']).strftime('%H:%M:%S'),
                    'Status': query_info['status'],
                    'Type': query_info.get('query_type', 'unknown'),
                    'Duration': f"{query_info.get('execution_time', 0):.2f}s",
                    'Query': query_info['query'][:50] + '...' if len(query_info['query']) > 50 else query_info['query']
                })

            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)

            # Re-run query from history
            selected_query = st.selectbox(
                "Re-run a previous query:",
                ["Select a query..."] + [q['query'] for q in reversed(st.session_state.query_history[-10:])]
            )

            if selected_query != "Select a query..." and st.button("🔄 Re-run Selected Query"):
                self.execute_query_with_enhanced_error_handling(selected_query)

    def render_data_explorer(self):
        """Enhanced data explorer interface"""
        st.header("📊 Enhanced Data Explorer")

        limits = st.session_state.query_limits

        # Quick table access
        st.subheader("🚀 Quick Table Access")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📋 List All Tables", key="list_all_tables"):
                self.list_tables()

        with col2:
            if st.button("🔄 Refresh Table Cache", key="refresh_cache"):
                self.refresh_table_cache()

        # Table selection and operations
        if st.session_state.tables_cache:
            st.subheader("📋 Available Tables")

            table_data = []
            for table in st.session_state.tables_cache:
                table_data.append({
                    'Table Name': table.get('TABLE_NAME', 'Unknown'),
                    'Schema': table.get('TABLE_SCHEMA', 'Unknown'),
                    'Estimated Rows': table.get('ROW_COUNT', 'Unknown')
                })

            tables_df = pd.DataFrame(table_data)
            st.dataframe(tables_df, use_container_width=True)

            # Table operations
            selected_table = st.selectbox(
                "Select a table to explore:",
                [""] + [table.get('TABLE_NAME', 'Unknown') for table in st.session_state.tables_cache]
            )

            if selected_table:
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    if st.button("👁️ Preview Data", key="preview_table"):
                        query = f"SELECT * FROM {selected_table} LIMIT {limits.default_preview}"
                        self.execute_query_with_enhanced_error_handling(query)

                with col2:
                    if st.button("📊 Get Statistics", key="table_stats"):
                        self.get_table_statistics(selected_table)

                with col3:
                    if st.button("🔍 Describe Structure", key="describe_table"):
                        query = f"DESCRIBE TABLE {selected_table}"
                        self.execute_query_with_enhanced_error_handling(query)

                with col4:
                    if st.button("📈 Sample Analysis", key="sample_analysis"):
                        query = f"SELECT * FROM {selected_table} LIMIT {limits.analytics_sample}"
                        self.execute_query_with_enhanced_error_handling(query)

        # Current data analysis
        if st.session_state.current_data is not None:
            st.divider()
            self.render_data_statistics()

    def render_admin_tools(self):
        """Enhanced admin tools interface"""
        st.header("⚙️ Enhanced Admin Tools")

        # System information
        st.subheader("🖥️ System Information")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Connection Status", st.session_state.get('connection_status', 'unknown'))

        with col2:
            st.metric("Total Queries", len(st.session_state.query_history))

        with col3:
            st.metric("Tables Cached", len(st.session_state.tables_cache))

        # Configuration management
        st.subheader("⚙️ Configuration Management")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Query Limits:**")
            limits = st.session_state.query_limits
            st.write(f"• Preview: {limits.default_preview:,} rows")
            st.write(f"• Search: {limits.default_search:,} rows")
            st.write(f"• Analytics: {limits.analytics_sample:,} rows")

            if st.button("🔄 Reset to Defaults"):
                st.session_state.query_limits = QueryLimits()
                st.success("Configuration reset to defaults!")
                st.rerun()

        with col2:
            st.write("**Session Management:**")

            if st.button("🗑️ Clear Query History"):
                st.session_state.query_history = []
                st.success("Query history cleared!")
                st.rerun()

            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.success("Chat history cleared!")
                st.rerun()

            if st.button("🗑️ Clear All Data"):
                for key in ['query_history', 'chat_history', 'current_data', 'tables_cache']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("All session data cleared!")
                st.rerun()

        # Export/Import configuration
        st.subheader("📤 Export/Import")

        if st.button("📤 Export Session Data"):
            session_data = {
                'query_history': st.session_state.get('query_history', []),
                'chat_history': st.session_state.get('chat_history', []),
                'query_limits': {
                    'default_preview': st.session_state.query_limits.default_preview,
                    'default_search': st.session_state.query_limits.default_search,
                    'analytics_sample': st.session_state.query_limits.analytics_sample
                },
                'export_timestamp': datetime.now().isoformat()
            }

            session_json = json.dumps(session_data, indent=2)
            st.download_button(
                "💾 Download Session Data",
                session_json,
                f"snowflake_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )

    def render_data_statistics(self):
        """Enhanced data statistics for current dataset"""
        if st.session_state.current_data is None:
            st.warning("⚠️ No data available for analysis. Please execute a query first.")
            return

        try:
            df = pd.DataFrame(st.session_state.current_data)
            if df.empty:
                st.warning("⚠️ The current dataset is empty.")
                return

            st.subheader("📊 Data Analysis Dashboard")

            # Basic metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                memory_usage = df.memory_usage(deep=True).sum() / 1024
                st.metric("Memory Usage", f"{memory_usage:.1f} KB")
            with col4:
                numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                st.metric("Numeric Columns", numeric_cols)

            # Data quality assessment
            st.subheader("🔍 Data Quality Overview")

            quality_data = []
            for col in df.columns:
                null_count = df[col].isnull().sum()
                null_pct = (null_count / len(df)) * 100 if len(df) > 0 else 0
                unique_count = df[col].nunique()
                unique_pct = (unique_count / len(df)) * 100 if len(df) > 0 else 0

                quality_data.append({
                    'Column': col,
                    'Data Type': str(df[col].dtype),
                    'Null Count': null_count,
                    'Null %': f"{null_pct:.1f}%",
                    'Unique Values': unique_count,
                    'Unique %': f"{unique_pct:.1f}%"
                })

            quality_df = pd.DataFrame(quality_data)
            st.dataframe(quality_df, use_container_width=True)

            # Quick visualizations
            self.render_quick_visualizations(df)

        except Exception as e:
            st.error(f"Error generating statistics: {str(e)}")

    def render_quick_visualizations(self, df):
        """Render quick visualizations for the dataset"""
        st.subheader("📈 Quick Visualizations")

        # Detect column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        if numeric_cols:
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Numerical Data Distribution**")
                selected_numeric = st.selectbox("Select numeric column:", numeric_cols, key="viz_numeric")

                if selected_numeric:
                    fig = px.histogram(df, x=selected_numeric, title=f"Distribution of {selected_numeric}")
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                if len(numeric_cols) >= 2:
                    st.write("**Correlation Analysis**")
                    corr_matrix = df[numeric_cols].corr()
                    fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Matrix")
                    st.plotly_chart(fig, use_container_width=True)

        if categorical_cols:
            st.write("**Categorical Data Analysis**")
            selected_categorical = st.selectbox("Select categorical column:", categorical_cols, key="viz_categorical")

            if selected_categorical:
                value_counts = df[selected_categorical].value_counts().head(10)
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                             title=f"Top 10 Values in {selected_categorical}")
                fig.update_xaxes(title=selected_categorical)
                fig.update_yaxes(title="Count")
                st.plotly_chart(fig, use_container_width=True)

    def render_export_options(self, df):
        """Render data export options"""
        st.subheader("💾 Export Current Dataset")

        col1, col2, col3 = st.columns(3)

        with col1:
            csv_data = df.to_csv(index=False)
            st.download_button(
                "📄 Download CSV",
                csv_data,
                f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )

        with col2:
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                "📋 Download JSON",
                json_data,
                f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )

        with col3:
            # Excel export would require openpyxl
            try:
                excel_buffer = io.BytesIO()
                df.to_excel(excel_buffer, index=False)
                excel_data = excel_buffer.getvalue()

                st.download_button(
                    "📊 Download Excel",
                    excel_data,
                    f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except ImportError:
                st.info("📊 Excel export requires openpyxl package")

    def list_tables(self):
        """List available tables"""
        try:
            with st.spinner("Fetching tables..."):
                result = self.mcp_client.execute_query("SHOW TABLES")
                result_data = json.loads(result)

                if result_data.get('status') == 'success' and result_data.get('data'):
                    st.session_state.tables_cache = result_data['data']

                    df = pd.DataFrame(result_data['data'])
                    st.success(f"✅ Found {len(df)} tables")
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("No tables found or query failed")
        except Exception as e:
            st.error(f"Error listing tables: {str(e)}")

    def refresh_table_cache(self):
        """Refresh the table cache"""
        try:
            with st.spinner("Refreshing table cache..."):
                result = self.mcp_client.execute_query("SHOW TABLES")
                result_data = json.loads(result)

                if result_data.get('status') == 'success' and result_data.get('data'):
                    st.session_state.tables_cache = result_data['data']
                    st.success(f"✅ Refreshed cache with {len(result_data['data'])} tables")
                else:
                    st.warning("Failed to refresh table cache")
        except Exception as e:
            st.error(f"Error refreshing cache: {str(e)}")

    def get_table_statistics(self, table_name: str):
        """Get basic statistics for a table"""
        try:
            queries = [
                f"SELECT COUNT(*) as row_count FROM {table_name}",
                f"SELECT * FROM {table_name} LIMIT 5"
            ]

            for query in queries:
                st.info(f"Executing: {query}")
                self.execute_query_with_enhanced_error_handling(query)

        except Exception as e:
            st.error(f"Error getting table statistics: {str(e)}")

    def test_connection(self):
        """Test the Snowflake connection"""
        try:
            with st.spinner("Testing connection..."):
                result = self.mcp_client.test_connection()
                result_data = json.loads(result)

                if result_data.get('status') == 'success':
                    st.session_state.connection_status = 'connected'
                    st.success("✅ Connection successful!")

                    # Display connection details
                    st.json(result_data)
                else:
                    st.session_state.connection_status = 'failed'
                    st.error("❌ Connection failed!")
        except Exception as e:
            st.session_state.connection_status = 'error'
            st.error(f"Connection error: {str(e)}")

    def reset_connection(self):
        """Reset the connection"""
        st.session_state.connection_status = 'disconnected'
        st.info("🔄 Connection reset. Use 'Test Connection' to reconnect.")

    def run(self):
        """Main application runner with enhanced tabbed interface"""
        # Main title
        st.title("❄️ Enhanced Snowflake AI Assistant")
        st.markdown("*Powered by advanced query management and AI capabilities*")

        # Render enhanced sidebar
        self.render_enhanced_sidebar()

        # Main tabbed interface
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏠 Dashboard",
            "🤖 AI Chat",
            "🔍 Query Tools",
            "📊 Data Explorer",
            "⚙️ Admin Tools"
        ])

        with tab1:
            self.render_dashboard()

        with tab2:
            self.render_ai_chat()

        with tab3:
            self.render_query_tools()

        with tab4:
            self.render_data_explorer()

        with tab5:
            self.render_admin_tools()


# Run the enhanced application
if __name__ == "__main__":
    try:
        app = EnhancedSnowflakeApp()
        app.run()
    except Exception as e:
        st.error(f"❌ Application failed to start: {str(e)}")
        st.error("Please check your configuration and try again.")

        # Debug information
        with st.expander("🔍 Debug Information"):
            st.text(f"Error type: {type(e).__name__}")
            st.text(f"Error details: {str(e)}")

            # Show system information
            st.subheader("System Information")
            st.text(f"Streamlit version: {st.__version__}")

            # Check required modules
            required_modules = ['pandas', 'numpy', 'plotly', 'json', 'datetime']
            for module in required_modules:
                try:
                    __import__(module)
                    st.success(f"✅ {module}: Available")
                except ImportError:
                    st.error(f"❌ {module}: Not Available")
