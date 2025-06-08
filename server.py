"""
Enhanced Snowflake MCP Server
Compatible with the Enhanced Streamlit AI Assistant

This server provides all the methods expected by the enhanced Streamlit app
including table management, data exploration, and performance monitoring.
"""

import json
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Snowflake connector
try:
    import snowflake.connector
    from snowflake.connector import DictCursor

    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False
    print("⚠️ Snowflake connector not available. Install with: pip install snowflake-connector-python")


class EnhancedSnowflakeServer:
    """
    Enhanced Snowflake MCP Server with comprehensive functionality
    Compatible with the Enhanced Streamlit AI Assistant
    """

    def __init__(self, connection_params: Optional[Dict[str, str]] = None):
        """
        Initialize the Enhanced Snowflake Server

        Args:
            connection_params: Dictionary with Snowflake connection parameters
                - account: Snowflake account identifier
                - user: Username
                - password: Password
                - warehouse: Warehouse name
                - database: Database name
                - schema: Schema name
                - role: Role (optional)
        """
        self.connection_params = connection_params or {}
        self.connection = None
        self.connected = False

        # Performance tracking
        self.query_count = 0
        self.total_execution_time = 0.0
        self.last_query_time = None

        # Cache for frequently accessed data
        self.tables_cache = []
        self.databases_cache = []
        self.warehouses_cache = []
        self.cache_timestamp = None
        self.cache_ttl = 300  # 5 minutes cache TTL

        logger.info("Enhanced Snowflake Server initialized")

    def _get_connection(self):
        """Get or create Snowflake connection"""
        if not SNOWFLAKE_AVAILABLE:
            raise Exception("Snowflake connector not available. Please install snowflake-connector-python")

        if self.connection is None or not self.connected:
            try:
                self.connection = snowflake.connector.connect(
                    account=self.connection_params.get('account'),
                    user=self.connection_params.get('user'),
                    password=self.connection_params.get('password'),
                    warehouse=self.connection_params.get('warehouse'),
                    database=self.connection_params.get('database'),
                    schema=self.connection_params.get('schema'),
                    role=self.connection_params.get('role'),
                    client_session_keep_alive=True
                )
                self.connected = True
                logger.info("Successfully connected to Snowflake")
            except Exception as e:
                self.connected = False
                logger.error(f"Failed to connect to Snowflake: {e}")
                raise

        return self.connection

    def _execute_query_internal(self, query: str, timeout: int = 300) -> Dict[str, Any]:
        """
        Internal method to execute queries and return structured results

        Args:
            query: SQL query to execute
            timeout: Query timeout in seconds

        Returns:
            Dictionary with query results and metadata
        """
        start_time = time.time()

        try:
            conn = self._get_connection()
            cursor = conn.cursor(DictCursor)

            # Set query timeout
            cursor.execute(f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {timeout}")

            # Execute the main query
            cursor.execute(query)

            # Fetch results
            results = cursor.fetchall()
            execution_time = time.time() - start_time

            # Update performance tracking
            self.query_count += 1
            self.total_execution_time += execution_time
            self.last_query_time = execution_time

            # Convert results to proper format
            data = []
            if results:
                for row in results:
                    # Convert any non-serializable types
                    clean_row = {}
                    for key, value in row.items():
                        if isinstance(value, (pd.Timestamp, np.datetime64)):
                            clean_row[key] = str(value)
                        elif isinstance(value, (np.integer, np.floating)):
                            clean_row[key] = value.item()
                        elif value is None:
                            clean_row[key] = None
                        else:
                            clean_row[key] = value
                    data.append(clean_row)

            cursor.close()

            return {
                "status": "success",
                "data": data,
                "row_count": len(data),
                "execution_time": execution_time,
                "query": query,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)

            logger.error(f"Query execution failed: {error_msg}")

            return {
                "status": "error",
                "error": error_msg,
                "execution_time": execution_time,
                "query": query,
                "timestamp": datetime.now().isoformat()
            }

    def execute_query(self, query: str, timeout: int = 300) -> str:
        """
        Execute SQL query and return JSON result
        Compatible with original Streamlit app

        Args:
            query: SQL query to execute
            timeout: Query timeout in seconds

        Returns:
            JSON string with query results
        """
        try:
            result = self._execute_query_internal(query, timeout)
            return json.dumps(result)
        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e),
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
            return json.dumps(error_result)

    def test_connection(self) -> str:
        """
        Test the Snowflake connection
        Returns JSON string with connection status
        """
        try:
            # Test with a simple query
            result = self._execute_query_internal(
                "SELECT CURRENT_VERSION() as version, CURRENT_WAREHOUSE() as warehouse, CURRENT_DATABASE() as database, CURRENT_SCHEMA() as schema")

            if result.get('status') == 'success' and result.get('data'):
                connection_info = result['data'][0]

                return json.dumps({
                    "status": "success",
                    "message": "Connected to Snowflake successfully",
                    "warehouse": connection_info.get('WAREHOUSE'),
                    "database": connection_info.get('DATABASE'),
                    "schema": connection_info.get('SCHEMA'),
                    "version": connection_info.get('VERSION'),
                    "timestamp": datetime.now().isoformat()
                })
            else:
                return json.dumps({
                    "status": "error",
                    "message": "Connection test query failed",
                    "timestamp": datetime.now().isoformat()
                })

        except Exception as e:
            self.connected = False
            return json.dumps({
                "status": "error",
                "message": f"Connection failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })

    def reset_connection(self) -> str:
        """
        Reset the connection
        Returns JSON string with reset status
        """
        try:
            if self.connection:
                self.connection.close()

            self.connection = None
            self.connected = False

            return json.dumps({
                "status": "success",
                "message": "Connection reset successfully",
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Reset failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })

    def get_connection_info(self) -> str:
        """
        Get current connection information
        Returns JSON string with connection details
        """
        try:
            if not self.connected:
                return json.dumps({
                    "status": "error",
                    "message": "Not connected to Snowflake",
                    "timestamp": datetime.now().isoformat()
                })

            # Get current session info
            result = self._execute_query_internal("""
                SELECT 
                    CURRENT_ACCOUNT() as account,
                    CURRENT_USER() as user,
                    CURRENT_WAREHOUSE() as warehouse,
                    CURRENT_DATABASE() as database,
                    CURRENT_SCHEMA() as schema,
                    CURRENT_ROLE() as role,
                    CURRENT_REGION() as region
            """)

            if result.get('status') == 'success' and result.get('data'):
                connection_info = result['data'][0]

                return json.dumps({
                    "status": "success",
                    "connection_info": {
                        "account": connection_info.get('ACCOUNT'),
                        "user": connection_info.get('USER'),
                        "warehouse": connection_info.get('WAREHOUSE'),
                        "database": connection_info.get('DATABASE'),
                        "schema": connection_info.get('SCHEMA'),
                        "role": connection_info.get('ROLE'),
                        "region": connection_info.get('REGION')
                    },
                    "timestamp": datetime.now().isoformat()
                })
            else:
                return json.dumps({
                    "status": "error",
                    "message": "Failed to get connection info",
                    "timestamp": datetime.now().isoformat()
                })

        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Failed to get connection info: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })

    def list_tables(self, database: Optional[str] = None, schema: Optional[str] = None,
                    pattern: Optional[str] = None, include_stats: bool = False) -> str:
        """
        List tables with optional filtering
        Returns JSON string with table list
        """
        try:
            # Check cache first
            current_time = time.time()
            if (self.tables_cache and self.cache_timestamp and
                    current_time - self.cache_timestamp < self.cache_ttl):
                return json.dumps({
                    "status": "success",
                    "tables": self.tables_cache,
                    "count": len(self.tables_cache),
                    "cached": True,
                    "timestamp": datetime.now().isoformat()
                })

            # Build the SHOW TABLES query
            query = "SHOW TABLES"

            # Add optional filters
            if database:
                query += f" IN DATABASE {database}"
            elif schema:
                query += f" IN SCHEMA {schema}"

            if pattern:
                query += f" LIKE '{pattern}'"

            # Execute the query
            result = self._execute_query_internal(query)

            if result.get('status') == 'success':
                tables_data = result.get('data', [])

                # If include_stats is True, get additional statistics
                if include_stats and tables_data:
                    enhanced_tables = []
                    for table in tables_data[:10]:  # Limit to first 10 for performance
                        table_name = table.get('name') or table.get('TABLE_NAME')
                        if table_name:
                            try:
                                # Get row count
                                count_result = self._execute_query_internal(
                                    f"SELECT COUNT(*) as row_count FROM {table_name}")
                                if count_result.get('status') == 'success' and count_result.get('data'):
                                    table['ROW_COUNT'] = count_result['data'][0].get('ROW_COUNT', 0)
                            except:
                                table['ROW_COUNT'] = 'Unknown'

                        enhanced_tables.append(table)

                    tables_data = enhanced_tables

                # Update cache
                self.tables_cache = tables_data
                self.cache_timestamp = current_time

                return json.dumps({
                    "status": "success",
                    "tables": tables_data,
                    "count": len(tables_data),
                    "cached": False,
                    "timestamp": datetime.now().isoformat()
                })
            else:
                return json.dumps({
                    "status": "error",
                    "message": f"Failed to list tables: {result.get('error', 'Unknown error')}",
                    "timestamp": datetime.now().isoformat()
                })

        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Failed to list tables: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })

    def describe_table(self, table_name: str, include_sample: bool = False) -> str:
        """
        Describe table structure with optional sample data
        Returns JSON string with table description
        """
        try:
            # Get table structure
            describe_query = f"DESCRIBE TABLE {table_name}"
            describe_result = self._execute_query_internal(describe_query)

            response = {
                "status": "success",
                "table_name": table_name,
                "timestamp": datetime.now().isoformat()
            }

            if describe_result.get('status') == 'success':
                response["columns"] = describe_result.get('data', [])

                # Get table info (row count, etc.)
                try:
                    info_queries = [
                        f"SELECT COUNT(*) as row_count FROM {table_name}",
                        f"SELECT COUNT(*) as column_count FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name.split('.')[-1]}'"
                    ]

                    table_info = {}
                    for query in info_queries:
                        try:
                            info_result = self._execute_query_internal(query)
                            if info_result.get('status') == 'success' and info_result.get('data'):
                                table_info.update(info_result['data'][0])
                        except:
                            continue

                    if table_info:
                        response["table_info"] = table_info

                except Exception as e:
                    logger.warning(f"Failed to get table info for {table_name}: {e}")

                # Include sample data if requested
                if include_sample:
                    try:
                        sample_query = f"SELECT * FROM {table_name} LIMIT 5"
                        sample_result = self._execute_query_internal(sample_query)

                        if sample_result.get('status') == 'success':
                            response["sample_data"] = sample_result.get('data', [])
                    except Exception as e:
                        logger.warning(f"Failed to get sample data for {table_name}: {e}")
            else:
                response["status"] = "error"
                response["message"] = f"Failed to describe table: {describe_result.get('error', 'Unknown error')}"

            return json.dumps(response)

        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Failed to describe table: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })

    def show_databases(self) -> str:
        """
        Show available databases
        Returns JSON string with database list
        """
        try:
            result = self._execute_query_internal("SHOW DATABASES")

            if result.get('status') == 'success':
                databases_data = result.get('data', [])
                self.databases_cache = databases_data

                return json.dumps({
                    "status": "success",
                    "databases": databases_data,
                    "count": len(databases_data),
                    "timestamp": datetime.now().isoformat()
                })
            else:
                return json.dumps({
                    "status": "error",
                    "message": "Failed to show databases",
                    "timestamp": datetime.now().isoformat()
                })

        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Failed to show databases: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })

    def show_warehouses(self) -> str:
        """
        Show available warehouses
        Returns JSON string with warehouse list
        """
        try:
            result = self._execute_query_internal("SHOW WAREHOUSES")

            if result.get('status') == 'success':
                warehouses_data = result.get('data', [])
                self.warehouses_cache = warehouses_data

                return json.dumps({
                    "status": "success",
                    "warehouses": warehouses_data,
                    "count": len(warehouses_data),
                    "timestamp": datetime.now().isoformat()
                })
            else:
                return json.dumps({
                    "status": "error",
                    "message": "Failed to show warehouses",
                    "timestamp": datetime.now().isoformat()
                })

        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Failed to show warehouses: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })

    def get_performance_stats(self) -> str:
        """
        Get performance statistics
        Returns JSON string with performance metrics
        """
        try:
            # Server-level stats
            server_stats = {
                "total_queries": self.query_count,
                "total_execution_time": self.total_execution_time,
                "average_execution_time": self.total_execution_time / max(self.query_count, 1),
                "last_query_time": self.last_query_time,
                "connection_status": "connected" if self.connected else "disconnected"
            }

            # Try to get Snowflake query history stats
            snowflake_stats = {}
            try:
                stats_query = """
                    SELECT 
                        COUNT(*) as total_queries_1h,
                        AVG(EXECUTION_TIME) as avg_execution_time_1h,
                        MAX(EXECUTION_TIME) as max_execution_time_1h,
                        MIN(EXECUTION_TIME) as min_execution_time_1h
                    FROM INFORMATION_SCHEMA.QUERY_HISTORY 
                    WHERE START_TIME >= DATEADD(hour, -1, CURRENT_TIMESTAMP())
                """

                result = self._execute_query_internal(stats_query)
                if result.get('status') == 'success' and result.get('data'):
                    snowflake_stats = result['data'][0]

            except Exception as e:
                logger.warning(f"Could not get Snowflake query history stats: {e}")

            return json.dumps({
                "status": "success",
                "performance_stats": {
                    "server_stats": server_stats,
                    "snowflake_stats": snowflake_stats
                },
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Failed to get performance stats: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })

    def switch_authentication_mode(self, use_oauth: bool = False) -> str:
        """
        Switch authentication mode (placeholder for future enhancement)
        Returns JSON string with switch status
        """
        try:
            auth_mode = "OAuth" if use_oauth else "Password"

            # This is a placeholder - actual implementation would depend on your auth setup
            logger.info(f"Authentication mode switch requested: {auth_mode}")

            return json.dumps({
                "status": "success",
                "message": f"Authentication mode set to {auth_mode}",
                "auth_mode": auth_mode,
                "note": "This is a placeholder implementation",
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Failed to switch authentication: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })

    def insert_csv_data(self, table_name: str, csv_content: str,
                        primary_key: Optional[str] = None, create_table: bool = True) -> str:
        """
        Insert CSV data into a table
        Returns JSON string with insert status
        """
        try:
            # Parse CSV content
            from io import StringIO
            df = pd.read_csv(StringIO(csv_content))

            if df.empty:
                return json.dumps({
                    "status": "error",
                    "message": "CSV content is empty",
                    "timestamp": datetime.now().isoformat()
                })

            # Create table if requested
            if create_table:
                # Generate CREATE TABLE statement based on DataFrame
                columns = []
                for col in df.columns:
                    # Simple type inference
                    if df[col].dtype in ['int64', 'int32']:
                        col_type = "INTEGER"
                    elif df[col].dtype in ['float64', 'float32']:
                        col_type = "FLOAT"
                    else:
                        col_type = "VARCHAR(255)"

                    pk_clause = " PRIMARY KEY" if col == primary_key else ""
                    columns.append(f"{col} {col_type}{pk_clause}")

                create_query = f"CREATE OR REPLACE TABLE {table_name} ({', '.join(columns)})"
                create_result = self._execute_query_internal(create_query)

                if create_result.get('status') != 'success':
                    return json.dumps({
                        "status": "error",
                        "message": f"Failed to create table: {create_result.get('error')}",
                        "timestamp": datetime.now().isoformat()
                    })

            # Insert data (this is a simplified implementation)
            # In a real scenario, you'd want to use Snowflake's COPY INTO command
            rows_inserted = 0
            for _, row in df.iterrows():
                values = [f"'{str(val)}'" if not pd.isna(val) else "NULL" for val in row]
                insert_query = f"INSERT INTO {table_name} VALUES ({', '.join(values)})"

                insert_result = self._execute_query_internal(insert_query)
                if insert_result.get('status') == 'success':
                    rows_inserted += 1

            return json.dumps({
                "status": "success",
                "message": f"CSV data inserted into {table_name}",
                "rows_inserted": rows_inserted,
                "total_rows": len(df),
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Failed to insert CSV data: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })

    def create_table_from_data(self, table_name: str, json_data: str,
                               primary_key: Optional[str] = None) -> str:
        """
        Create table from JSON data
        Returns JSON string with creation status
        """
        try:
            # Parse JSON data
            data = json.loads(json_data)

            if not data or not isinstance(data, list):
                return json.dumps({
                    "status": "error",
                    "message": "JSON data must be a non-empty list",
                    "timestamp": datetime.now().isoformat()
                })

            # Infer schema from first record
            first_record = data[0]
            columns = []

            for key, value in first_record.items():
                if isinstance(value, int):
                    col_type = "INTEGER"
                elif isinstance(value, float):
                    col_type = "FLOAT"
                elif isinstance(value, bool):
                    col_type = "BOOLEAN"
                else:
                    col_type = "VARCHAR(255)"

                pk_clause = " PRIMARY KEY" if key == primary_key else ""
                columns.append(f"{key} {col_type}{pk_clause}")

            create_query = f"CREATE OR REPLACE TABLE {table_name} ({', '.join(columns)})"
            result = self._execute_query_internal(create_query)

            if result.get('status') == 'success':
                return json.dumps({
                    "status": "success",
                    "message": f"Table {table_name} created successfully",
                    "columns_created": len(columns),
                    "timestamp": datetime.now().isoformat()
                })
            else:
                return json.dumps({
                    "status": "error",
                    "message": f"Failed to create table: {result.get('error')}",
                    "timestamp": datetime.now().isoformat()
                })

        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Failed to create table: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })

    def upsert_data(self, table_name: str, json_data: str, primary_key: str) -> str:
        """
        Upsert JSON data into table
        Returns JSON string with upsert status
        """
        try:
            # Parse JSON data
            data = json.loads(json_data)

            if not data:
                return json.dumps({
                    "status": "error",
                    "message": "JSON data is empty",
                    "timestamp": datetime.now().isoformat()
                })

            # Simple upsert implementation using MERGE
            # In practice, you'd want to optimize this for larger datasets
            rows_affected = 0

            for record in data:
                if not isinstance(record, dict) or primary_key not in record:
                    continue

                # Build column names and values
                columns = list(record.keys())
                values = [f"'{str(val)}'" if val is not None else "NULL" for val in record.values()]

                # Create a simple MERGE statement
                merge_query = f"""
                MERGE INTO {table_name} AS target
                USING (SELECT {', '.join([f'{val} AS {col}' for col, val in zip(columns, values)])}) AS source
                ON target.{primary_key} = source.{primary_key}
                WHEN MATCHED THEN UPDATE SET {', '.join([f'{col} = source.{col}' for col in columns if col != primary_key])}
                WHEN NOT MATCHED THEN INSERT ({', '.join(columns)}) VALUES ({', '.join(values)})
                """

                result = self._execute_query_internal(merge_query)
                if result.get('status') == 'success':
                    rows_affected += 1

            return json.dumps({
                "status": "success",
                "message": f"Data upserted into {table_name}",
                "rows_affected": rows_affected,
                "total_records": len(data),
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Failed to upsert data: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })

    def close(self):
        """Close the connection"""
        if self.connection:
            self.connection.close()
            self.connected = False
            logger.info("Snowflake connection closed")


# Example usage and configuration
if __name__ == "__main__":
    # Example connection parameters
    # Replace these with your actual Snowflake credentials
    connection_params = {
        'account': 'your_account.region',  # e.g., 'abc12345.us-east-1'
        'user': 'your_username',
        'password': 'your_password',
        'warehouse': 'your_warehouse',
        'database': 'your_database',
        'schema': 'your_schema',
        'role': 'your_role'  # optional
    }

    # Create server instance
    server = EnhancedSnowflakeServer(connection_params)

    # Test basic functionality
    try:
        print("Testing connection...")
        result = server.test_connection()
        print(f"Connection test result: {result}")

        print("\nListing tables...")
        tables_result = server.list_tables()
        print(f"Tables result: {tables_result}")

        print("\nGetting performance stats...")
        perf_result = server.get_performance_stats()
        print(f"Performance stats: {perf_result}")

    except Exception as e:
        print(f"Error during testing: {e}")

    finally:
        server.close()
