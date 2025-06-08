#!/usr/bin/env python3
"""
Comprehensive Test Framework for Snowflake MCP Server
Save this as test_framework.py in your project directory
"""

import unittest
import json
import os
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from snowflake_bridge import EnhancedSnowflakeServer as BridgeServer

    BRIDGE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Could not import bridge: {e}")
    BRIDGE_AVAILABLE = False


class TestSnowflakeConnection(unittest.TestCase):
    """Test connection functionality without requiring actual Snowflake credentials"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = {
            'account': 'test_account.region',
            'user': 'test_user',
            'password': 'test_password',
            'warehouse': 'TEST_WH',
            'database': 'TEST_DB',
            'schema': 'TEST_SCHEMA',
            'role': 'TEST_ROLE'
        }

    @unittest.skipUnless(BRIDGE_AVAILABLE, "Bridge server not available")
    def test_bridge_initialization_no_config(self):
        """Test bridge initialization without config"""
        server = BridgeServer()
        self.assertIsNotNone(server)
        self.assertEqual(server.max_rows, 1000)

    @unittest.skipUnless(BRIDGE_AVAILABLE, "Bridge server not available")
    def test_bridge_config_loading(self):
        """Test bridge config loading from file"""
        # Create temporary config file
        config_data = {"snowflake": self.mock_config}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            server = BridgeServer(config_path)
            self.assertTrue(server._load_config())
            self.assertEqual(server.config.get('account'), 'test_account.region')
        finally:
            os.unlink(config_path)

    def test_bridge_environment_config(self):
        """Test bridge environment variable configuration"""
        if not BRIDGE_AVAILABLE:
            self.skipTest("Bridge server not available")

        # Mock environment variables
        env_vars = {
            'SNOWFLAKE_ACCOUNT': 'env_account',
            'SNOWFLAKE_USER': 'env_user',
            'SNOWFLAKE_PASSWORD': 'env_password',
            'SNOWFLAKE_DATABASE': 'env_db',
            'SNOWFLAKE_SCHEMA': 'env_schema',
            'SNOWFLAKE_WAREHOUSE': 'env_wh'
        }

        with patch.dict(os.environ, env_vars):
            server = BridgeServer()
            self.assertTrue(server._load_config())
            self.assertEqual(server.config.get('account'), 'env_account')


class TestMockSnowflakeOperations(unittest.TestCase):
    """Test server operations with mocked Snowflake connections"""

    def setUp(self):
        """Set up mock connections"""
        self.mock_connection = Mock()
        self.mock_cursor = Mock()
        self.mock_connection.cursor.return_value = self.mock_cursor

    @unittest.skipUnless(BRIDGE_AVAILABLE, "Bridge server not available")
    @patch('snowflake.connector.connect')
    def test_bridge_test_connection(self, mock_connect):
        """Test bridge connection testing"""
        mock_connect.return_value = self.mock_connection
        self.mock_cursor.fetchone.return_value = ['7.0.0']

        server = BridgeServer()
        server.config = {'account': 'test', 'user': 'test', 'password': 'test'}
        result = server.test_connection()
        result_data = json.loads(result)

        self.assertEqual(result_data['status'], 'success')

    @unittest.skipUnless(BRIDGE_AVAILABLE, "Bridge server not available")
    @patch('snowflake.connector.connect')
    def test_bridge_query_execution(self, mock_connect):
        """Test bridge query execution"""
        mock_connect.return_value = self.mock_connection

        # Mock cursor description and results
        self.mock_cursor.description = [('ID',), ('NAME',), ('VALUE',)]
        self.mock_cursor.fetchall.return_value = [
            (1, 'Test 1', 100.0),
            (2, 'Test 2', 200.0)
        ]

        server = BridgeServer()
        server.config = {'account': 'test', 'user': 'test', 'password': 'test'}
        result = server.execute_query("SELECT * FROM test_table")
        result_data = json.loads(result)

        self.assertEqual(result_data['status'], 'success')
        self.assertEqual(len(result_data['data']), 2)

    def test_query_limit_addition(self):
        """Test automatic LIMIT addition for SELECT queries"""
        if not BRIDGE_AVAILABLE:
            self.skipTest("Bridge server not available")

        with patch('snowflake.connector.connect') as mock_connect:
            mock_connect.return_value = self.mock_connection
            self.mock_cursor.description = [('COUNT',)]
            self.mock_cursor.fetchall.return_value = [(10,)]

            server = BridgeServer()
            server.config = {'account': 'test', 'user': 'test', 'password': 'test'}
            server.max_rows = 500

            # Should add LIMIT
            server.execute_query("SELECT * FROM test_table")

            # Verify LIMIT was added
            calls = self.mock_cursor.execute.call_args_list
            query_call = None
            for call in calls:
                if 'SELECT' in str(call[0][0]):
                    query_call = call[0][0]
                    break

            self.assertIsNotNone(query_call)
            self.assertIn('LIMIT 500', query_call)


class TestErrorHandling(unittest.TestCase):
    """Test error handling scenarios"""

    @unittest.skipUnless(BRIDGE_AVAILABLE, "Bridge server not available")
    def test_missing_config_error(self):
        """Test behavior when config is missing"""
        server = BridgeServer()
        server.config = {}  # Empty config

        result = server.test_connection()
        result_data = json.loads(result)

        self.assertEqual(result_data['status'], 'error')
        self.assertIn('configuration', result_data['error'].lower())

    @unittest.skipUnless(BRIDGE_AVAILABLE, "Bridge server not available")
    @patch('snowflake.connector.connect')
    def test_connection_failure(self, mock_connect):
        """Test connection failure handling"""
        mock_connect.side_effect = Exception("Connection failed")

        server = BridgeServer()
        server.config = {'account': 'test', 'user': 'test', 'password': 'test'}

        result = server.test_connection()
        result_data = json.loads(result)

        self.assertEqual(result_data['status'], 'error')

    @unittest.skipUnless(BRIDGE_AVAILABLE, "Bridge server not available")
    @patch('snowflake.connector.connect')
    def test_query_execution_error(self, mock_connect):
        """Test query execution error handling"""
        mock_connect.return_value = self.mock_connection
        self.mock_cursor.execute.side_effect = Exception("Query failed")

        server = BridgeServer()
        server.config = {'account': 'test', 'user': 'test', 'password': 'test'}

        result = server.execute_query("INVALID SQL")
        result_data = json.loads(result)

        self.assertEqual(result_data['status'], 'error')

    def setUp(self):
        self.mock_connection = Mock()
        self.mock_cursor = Mock()
        self.mock_connection.cursor.return_value = self.mock_cursor


class IntegrationTestRunner:
    """Integration test runner for end-to-end testing"""

    def __init__(self, use_real_connection=False):
        self.use_real_connection = use_real_connection
        self.test_results = []

    def run_connectivity_tests(self):
        """Run connectivity tests"""
        print("\n🔗 Running Connectivity Tests...")

        if not BRIDGE_AVAILABLE:
            print("❌ Bridge server not available")
            return False

        try:
            server = BridgeServer()

            # Test config loading
            config_loaded = server._load_config()
            print(f"📁 Config loading: {'✅' if config_loaded else '❌'}")

            if self.use_real_connection and config_loaded:
                # Test actual connection
                result = server.test_connection()
                result_data = json.loads(result)
                success = result_data.get('status') == 'success'
                print(f"🔌 Real connection test: {'✅' if success else '❌'}")
                if not success:
                    print(f"   Error: {result_data.get('error', 'Unknown')}")
            else:
                print("🔌 Real connection test: ⏭️ Skipped (no config or disabled)")

            return True
        except Exception as e:
            print(f"❌ Connectivity tests failed: {e}")
            return False

    def run_functionality_tests(self):
        """Run functionality tests"""
        print("\n⚙️ Running Functionality Tests...")

        if not BRIDGE_AVAILABLE:
            print("❌ Bridge server not available")
            return False

        try:
            # Test with mock data
            with patch('snowflake.connector.connect') as mock_connect:
                mock_connection = Mock()
                mock_cursor = Mock()
                mock_connection.cursor.return_value = mock_cursor
                mock_connect.return_value = mock_connection

                server = BridgeServer()
                server.config = {'account': 'test', 'user': 'test', 'password': 'test'}

                # Test various query types
                test_cases = [
                    ("SELECT * FROM test", "SELECT query"),
                    ("SHOW TABLES", "SHOW query"),
                    ("DESCRIBE test_table", "DESCRIBE query")
                ]

                for query, description in test_cases:
                    mock_cursor.description = [('TEST_COL',)]
                    mock_cursor.fetchall.return_value = [('test_value',)]

                    result = server.execute_query(query)
                    result_data = json.loads(result)
                    success = result_data.get('status') == 'success'
                    print(f"🔍 {description}: {'✅' if success else '❌'}")

            return True
        except Exception as e:
            print(f"❌ Functionality tests failed: {e}")
            return False

    def run_streamlit_compatibility_tests(self):
        """Test compatibility with Streamlit app"""
        print("\n🎨 Running Streamlit Compatibility Tests...")

        try:
            # Test that the import works
            from snowflake_bridge import EnhancedSnowflakeServer
            print("📦 Import test: ✅")

            # Test initialization without errors
            server = EnhancedSnowflakeServer()
            print("🏗️ Initialization test: ✅")

            # Test required methods exist
            required_methods = [
                'test_connection',
                'execute_query',
                'list_tables',
                'describe_table'
            ]

            for method in required_methods:
                if hasattr(server, method):
                    print(f"🔧 Method '{method}': ✅")
                else:
                    print(f"🔧 Method '{method}': ❌")
                    return False

            return True
        except Exception as e:
            print(f"❌ Streamlit compatibility tests failed: {e}")
            return False

    def run_all_tests(self):
        """Run all integration tests"""
        print("🧪 Starting Integration Test Suite...")
        print("=" * 50)

        tests = [
            self.run_connectivity_tests,
            self.run_functionality_tests,
            self.run_streamlit_compatibility_tests
        ]

        passed = 0
        for test in tests:
            if test():
                passed += 1

        print("\n" + "=" * 50)
        print(f"📊 Integration Test Results: {passed}/{len(tests)} passed")

        if passed == len(tests):
            print("🎉 All integration tests passed!")
        else:
            print("⚠️ Some tests failed. Check the output above.")

        return passed == len(tests)


def main():
    """Main test runner"""
    print("🧪 Snowflake MCP Server Test Suite")
    print("=" * 50)

    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)

    print("\n" + "=" * 50)

    # Run integration tests
    integration_runner = IntegrationTestRunner(use_real_connection=False)
    integration_runner.run_all_tests()

    print("\n💡 To test with real Snowflake connection:")
    print("   1. Set up config.json or environment variables")
    print("   2. Run: python test_framework.py --real-connection")


if __name__ == "__main__":
    import sys

    use_real_connection = "--real-connection" in sys.argv

    if use_real_connection:
        print("🔴 Using real Snowflake connection for testing")
        integration_runner = IntegrationTestRunner(use_real_connection=True)
        integration_runner.run_all_tests()
    else:
        main()
