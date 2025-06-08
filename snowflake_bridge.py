"""
Improved Snowflake Bridge for Streamlit Integration
Addresses issues found in code review and adds missing functionality
"""

import json
import snowflake.connector
import pandas as pd
from typing import Optional, Dict, Any, List
import os
import time
from pathlib import Path
import logging
from datetime import datetime
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedSnowflakeServer:
    """
    Improved Snowflake Bridge with enhanced error handling,
    performance tracking, and comprehensive functionality
    """

    def __init__(self, config_path: Optional[str] = None, auto_connect: bool = False):
        self.connection = None
        self.config = {}
        self.config_path = config_path or self._find_config_file()
        self.max_rows = 1000
        self.auto_reconnect = True

        # Performance tracking
        self.query_count = 0
        self.total_execution_time = 0.0
        self.last_query_time = None
        self.connection_attempts = 0

        # Connection state
        self.connected = False
        self.last_connection_test = None

        # Load configuration
        self._load_config()

        # Auto-connect if requested and config is available
        if auto_connect and self.config:
            try:
                self._ensure_connection()
            except Exception as e:
                logger.warning(f"Auto-connect failed: {e}")

    def _find_config_file(self) -> Optional[str]:
        """Find config.json in various locations"""
        possible_paths = [
            "config.json",
            "../config.json",
            Path(__file__).parent / "config.json",
            Path.home() / ".snowflake" / "config.json"
        ]

        for path in possible_paths:
            if Path(path).exists():
                logger.info(f"Found config file: {path}")
                return str(path)

        logger.info("No config file found, will use environment variables")
        return None

    def _load_config(self) -> bool:
        """Load configuration from file or environment with validation"""
        try:
            # Try file first
            if self.config_path and Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                    self.config = file_config.get("snowflake", {})
                    logger.info(f"Loaded config from {self.config_path}")
            else:
                # Fallback to environment variables
                self.config = {
                    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
                    "user": os.getenv("SNOWFLAKE_USER"),
                    "password": os.getenv("SNOWFLAKE_PASSWORD"),
                    "database": os.getenv("SNOWFLAKE_DATABASE"),
                    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
                    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
                    "role": os.getenv("SNOWFLAKE_ROLE")
                }

                # Remove None values
                self.config = {k: v for k, v in self.config.items() if v is not None}
                logger.info("Using environment variable configuration")

            # Validate required fields
            required_fields = ["account", "user", "password"]
            missing_fields = [field for field in required_fields if not self.config.get(field)]

            if missing_fields:
                logger.error(f"Missing required configuration fields: {missing_fields}")
                return False

            # Add connection parameters for better reliability
            self.config.update({
                'client_session_keep_alive': True,
                'login_timeout': 60,
                'network_timeout': 60
            })

            return True

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False

    @contextmanager
    def _get_cursor(self):
        """Context manager for database cursors with proper cleanup"""
        cursor = None
        try:
            self._ensure_connection()
            cursor = self.connection.cursor()
            yield cursor
        except Exception as e:
            logger.error(f"Cursor operation failed: {e}")
            raise
        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception as e:
                    logger.warning(f"Failed to close cursor: {e}")

    def _ensure_connection(self):
        """Ensure we have an active connection with retry logic"""
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                if not self.connection or self.connection.is_closed():
                    if not self.config:
                        raise Exception("No configuration available for connection")

                    logger.info(f"Establishing connection (attempt {retry_count + 1})")
                    self.connection = snowflake.connector.connect(**self.config)
                    self.connected = True
                    self.connection_attempts += 1
                    logger.info("Successfully connected to Snowflake")
                    return

                # Test existing connection
                with self.connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                    return  # Connection is good

            except Exception as e:
                retry_count += 1
                self.connected = False
                logger.warning(f"Connection attempt {retry_count} failed: {e}")

                if retry_count < max_retries:
                    time.sleep(2 ** retry_count)  # Exponential backoff
                else:
                    raise Exception(f"Failed to connect after {max_retries} attempts: {e}")

    def _standardize_response(self, status: str, data: Any = None, error: str = None,
                              execution_time: float = None, query: str = None) -> str:
        """Standardize all response formats"""
        response = {
            "status": status,
            "timestamp": datetime.now().isoformat()
        }

        if data is not None:
            if isinstance(data, list):
                response["data"] = data
                response["row_count"] = len(data)
            else:
                response["data"] = data

        if error:
            response["error"] = error

        if execution_time is not None:
            response["execution_time"] = execution_time

        if query:
            response["query"] = query

        return json.dumps(response, default=str)

    def test_connection(self) -> str:
        """Test Snowflake connection with detailed diagnostics"""
        start_time = time.time()

        try:
            if not self.config:
                if not self._load_config():
                    return self._standardize_response(
                        "error",
                        error="Failed to load configuration. Please check config.json or environment variables."
                    )

            # Test connection with diagnostic query
            with self._get_cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        CURRENT_VERSION() as version,
                        CURRENT_WAREHOUSE() as warehouse,
                        CURRENT_DATABASE() as database,
                        CURRENT_SCHEMA() as schema,
                        CURRENT_USER() as user,
                        CURRENT_ROLE() as role
                """)
                result = cursor.fetchone()

                if result:
                    connection_info = {
                        "version": result[0],
                        "warehouse": result[1] or self.config.get("warehouse", "N/A"),
                        "database": result[2] or self.config.get("database", "N/A"),
                        "schema": result[3] or self.config.get("schema", "N/A"),
                        "user": result[4],
                        "role": result[5]
                    }

                    self.last_connection_test = datetime.now()
                    execution_time = time.time() - start_time

                    return self._standardize_response(
                        "success",
                        data={
                            "message": "Connected to Snowflake successfully",
                            "connection_info": connection_info
                        },
                        execution_time=execution_time
                    )

        except Exception as e:
            execution_time = time.time() - start_time
            self.connected = False
            return self._standardize_response(
                "error",
                error=str(e),
                execution_time=execution_time
            )

    def execute_query(self, query: str, timeout: int = 300) -> str:
        """Execute a query with enhanced error handling and performance tracking"""
        start_time = time.time()
        original_query = query.strip()

        try:
            # Validate query
            if not query.strip():
                return self._standardize_response("error", error="Empty query provided")

            with self._get_cursor() as cursor:
                # Set timeout
                cursor.execute(f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {timeout}")

                # Add limit for SELECT queries if not present
                query_upper = query.upper().strip()
                if (query_upper.startswith("SELECT") and
                        "LIMIT" not in query_upper and
                        "COUNT(" not in query_upper):
                    query = f"{query.rstrip(';')} LIMIT {self.max_rows}"
                    logger.info(f"Added LIMIT {self.max_rows} to query for performance")

                # Execute query
                cursor.execute(query)

                # Handle different query types
                if any(query_upper.startswith(cmd) for cmd in ["SELECT", "SHOW", "DESCRIBE", "DESC", "EXPLAIN"]):
                    results = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description] if cursor.description else []

                    # Convert to list of dicts with proper type handling
                    data = []
                    for row in results:
                        row_dict = {}
                        for i, col in enumerate(columns):
                            value = row[i] if i < len(row) else None

                            # Handle special types
                            if isinstance(value, (bytes, bytearray)):
                                row_dict[col] = value.decode('utf-8', errors='replace')
                            elif pd.isna(value):
                                row_dict[col] = None
                            elif isinstance(value, (pd.Timestamp,)):
                                row_dict[col] = value.isoformat()
                            else:
                                row_dict[col] = value
                        data.append(row_dict)

                    execution_time = time.time() - start_time

                    # Update performance metrics
                    self.query_count += 1
                    self.total_execution_time += execution_time
                    self.last_query_time = execution_time

                    return self._standardize_response(
                        "success",
                        data=data,
                        execution_time=execution_time,
                        query=original_query
                    )
                else:
                    # For non-SELECT queries (INSERT, UPDATE, DELETE, CREATE, etc.)
                    row_count = cursor.rowcount if hasattr(cursor, 'rowcount') else 0
                    execution_time = time.time() - start_time

                    # Update performance metrics
                    self.query_count += 1
                    self.total_execution_time += execution_time
                    self.last_query_time = execution_time

                    return self._standardize_response(
                        "success",
                        data={"message": "Query executed successfully", "affected_rows": row_count},
                        execution_time=execution_time,
                        query=original_query
                    )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)

            # Categorize common errors
            if "does not exist" in error_msg.lower():
                error_type = "Object not found"
            elif "permission" in error_msg.lower() or "access" in error_msg.lower():
                error_type = "Permission denied"
            elif "timeout" in error_msg.lower():
                error_type = "Query timeout"
            elif "syntax" in error_msg.lower():
                error_type = "Syntax error"
            else:
                error_type = "Execution error"

            logger.error(f"Query execution failed ({error_type}): {error_msg}")

            return self._standardize_response(
                "error",
                error=f"{error_type}: {error_msg}",
                execution_time=execution_time,
                query=original_query
            )

    def list_tables(self, database: str = None, schema: str = None,
                    pattern: str = None, include_stats: bool = False) -> str:
        """List tables with optional filtering and statistics"""
        try:
            # Build query
            if schema:
                query = f"SHOW TABLES IN SCHEMA {schema}"
            elif database:
                query = f"SHOW TABLES IN DATABASE {database}"
            else:
                query = "SHOW TABLES"

            if pattern:
                query += f" LIKE '{pattern}'"

            result = self.execute_query(query)
            result_data = json.loads(result)

            if result_data.get('status') == 'success' and include_stats:
                # Add row count statistics for first 10 tables
                tables = result_data.get('data', [])[:10]
                enhanced_tables = []

                for table in tables:
                    table_name = table.get('name') or table.get('TABLE_NAME')
                    if table_name:
                        try:
                            count_result = self.execute_query(f"SELECT COUNT(*) as ROW_COUNT FROM {table_name}")
                            count_data = json.loads(count_result)
                            if count_data.get('status') == 'success' and count_data.get('data'):
                                table['ESTIMATED_ROWS'] = count_data['data'][0].get('ROW_COUNT', 0)
                        except:
                            table['ESTIMATED_ROWS'] = 'Unknown'
                    enhanced_tables.append(table)

                result_data['data'] = enhanced_tables
                return json.dumps(result_data)

            return result

        except Exception as e:
            return self._standardize_response("error", error=f"Failed to list tables: {str(e)}")

    def describe_table(self, table_name: str, include_sample: bool = False) -> str:
        """Describe table structure with optional sample data"""
        try:
            # Get table structure
            result = self.execute_query(f"DESCRIBE TABLE {table_name}")
            result_data = json.loads(result)

            if result_data.get('status') == 'success' and include_sample:
                # Add sample data
                sample_result = self.execute_query(f"SELECT * FROM {table_name} LIMIT 5")
                sample_data = json.loads(sample_result)

                if sample_data.get('status') == 'success':
                    result_data['sample_data'] = sample_data.get('data', [])

            return json.dumps(result_data)

        except Exception as e:
            return self._standardize_response("error", error=f"Failed to describe table: {str(e)}")

    def show_databases(self) -> str:
        """Show available databases"""
        return self.execute_query("SHOW DATABASES")

    def show_warehouses(self) -> str:
        """Show available warehouses"""
        return self.execute_query("SHOW WAREHOUSES")

    def show_schemas(self, database: str = None) -> str:
        """Show available schemas"""
        if database:
            return self.execute_query(f"SHOW SCHEMAS IN DATABASE {database}")
        else:
            return self.execute_query("SHOW SCHEMAS")

    def get_performance_stats(self) -> str:
        """Get performance statistics"""
        try:
            stats = {
                "query_count": self.query_count,
                "total_execution_time": self.total_execution_time,
                "average_execution_time": self.total_execution_time / max(self.query_count, 1),
                "last_query_time": self.last_query_time,
                "connection_attempts": self.connection_attempts,
                "connection_status": "connected" if self.connected else "disconnected",
                "last_connection_test": self.last_connection_test.isoformat() if self.last_connection_test else None
            }

            return self._standardize_response("success", data=stats)

        except Exception as e:
            return self._standardize_response("error", error=f"Failed to get performance stats: {str(e)}")

    def reset_connection(self) -> str:
        """Reset the connection"""
        try:
            if self.connection:
                self.connection.close()

            self.connection = None
            self.connected = False

            return self._standardize_response("success", data={"message": "Connection reset successfully"})

        except Exception as e:
            return self._standardize_response("error", error=f"Reset failed: {str(e)}")

    def get_connection_info(self) -> str:
        """Get current connection information"""
        try:
            if not self.connected:
                return self._standardize_response("error", error="Not connected to Snowflake")

            info_query = """
                SELECT 
                    CURRENT_ACCOUNT() as account,
                    CURRENT_USER() as user,
                    CURRENT_WAREHOUSE() as warehouse,
                    CURRENT_DATABASE() as database,
                    CURRENT_SCHEMA() as schema,
                    CURRENT_ROLE() as role,
                    CURRENT_REGION() as region
            """

            result = self.execute_query(info_query)
            result_data = json.loads(result)

            if result_data.get('status') == 'success' and result_data.get('data'):
                return self._standardize_response(
                    "success",
                    data={"connection_info": result_data['data'][0]}
                )
            else:
                return self._standardize_response("error", error="Failed to get connection info")

        except Exception as e:
            return self._standardize_response("error", error=f"Failed to get connection info: {str(e)}")

    def close(self):
        """Close the connection and cleanup"""
        try:
            if self.connection:
                self.connection.close()
                self.connected = False
                logger.info("Snowflake connection closed")
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Backward compatibility alias
EnhancedSnowflakeServer = ImprovedSnowflakeServer

# Example usage
if __name__ == "__main__":
    # Test the improved server
    server = ImprovedSnowflakeServer()

    print("Testing connection...")
    result = server.test_connection()
    print(f"Connection result: {result}")

    if json.loads(result).get('status') == 'success':
        print("\nTesting query execution...")
        query_result = server.execute_query("SELECT CURRENT_VERSION()")
        print(f"Query result: {query_result}")

        print("\nGetting performance stats...")
        perf_result = server.get_performance_stats()
        print(f"Performance stats: {perf_result}")

    server.close()
