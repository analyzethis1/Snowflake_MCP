#!/bin/bash

# Snowflake MCP Server with AI - macOS Launcher
# Enhanced setup script for Streamlit-based Snowflake data exploration tool

# Step 1: Move into the directory where the .command file itself is located
cd "$(dirname "$0")"

# Step 2: Setup output and log files
OUTPUT_DIR="output"
LOG_FILE="snowflake_mcp_server.log"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Step 3: Display banner
echo "❄️ =============================================="
echo "   Snowflake MCP Server with AI Assistant"
echo "   Streamlit-Based Data Exploration Tool"
echo "=============================================="
echo ""

# Step 4: Setup pyenv and Python environment
echo "🐍 Setting up Python environment..."

# Initialize pyenv if it exists
if command -v pyenv >/dev/null 2>&1; then
    echo "✅ Found pyenv - initializing..."
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"

    # Try to set a working Python version
    AVAILABLE_VERSIONS=$(pyenv versions --bare | grep -E "^3\.(9|10|11|12)" | head -1)
    if [ -n "$AVAILABLE_VERSIONS" ]; then
        echo "🔧 Setting Python version to: $AVAILABLE_VERSIONS"
        pyenv local "$AVAILABLE_VERSIONS"
    fi
else
    echo "ℹ️ pyenv not found - using system Python"
fi

# Step 5: Find working Python command
PYTHON_CMD=""
echo "🔍 Searching for Python installation..."

# Try different Python commands in order of preference
for cmd in python3 python3.12 python3.11 python3.10 python3.9 /opt/homebrew/bin/python3 /usr/local/bin/python3 /usr/bin/python3; do
    if command -v "$cmd" >/dev/null 2>&1; then
        # Test if the command actually works
        if $cmd --version >/dev/null 2>&1; then
            PYTHON_VERSION=$($cmd --version 2>&1)
            echo "✅ Found working Python: $PYTHON_VERSION at $cmd"
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "❌ No working Python 3 installation found."
    echo ""
    echo "🔧 Please install Python 3.9+ using one of these methods:"
    echo "   1. Homebrew: brew install python@3.11"
    echo "   2. pyenv: pyenv install 3.11.7 && pyenv global 3.11.7"
    echo "   3. Official installer: https://python.org/downloads/"
    echo ""
    read -p "Press any key to exit..." -n1 -s
    exit 1
fi

# Step 6: Check and setup virtual environment
VENV_NAME=".venv"
if [ ! -d "$VENV_NAME" ]; then
    echo "🔧 Virtual environment not found. Creating '$VENV_NAME'..."

    $PYTHON_CMD -m venv "$VENV_NAME"
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        echo "🔧 Trying alternative method..."

        # Try installing venv if it's missing
        echo "📦 Installing python3-venv..."
        if command -v brew >/dev/null 2>&1; then
            brew install python@3.11
        fi

        # Retry with explicit venv creation
        $PYTHON_CMD -m pip install --user virtualenv
        $PYTHON_CMD -m virtualenv "$VENV_NAME"

        if [ $? -ne 0 ]; then
            echo "❌ Still failed to create virtual environment"
            echo "🔧 Please check your Python installation"
            read -p "Press any key to exit..." -n1 -s
            exit 1
        fi
    fi
fi

echo "✅ Activating virtual environment..."
source "$VENV_NAME/bin/activate"

# Verify activation worked
if [ "$VIRTUAL_ENV" != "" ]; then
    echo "✅ Virtual environment activated: $VIRTUAL_ENV"
else
    echo "⚠️ Virtual environment activation may have failed"
fi

# Step 7: Upgrade pip and setuptools
echo "📦 Upgrading pip and setuptools..."
python -m pip install --upgrade pip setuptools wheel --quiet

# Step 8: Install dependencies from requirements.txt (if exists)
if [ -f "requirements.txt" ]; then
    echo "📋 Installing dependencies from requirements.txt..."
    python -m pip install -r requirements.txt --quiet
else
    # Step 9: Install core dependencies manually
    echo "📦 Installing core Python packages..."

    # Core web framework
    echo "  📱 Installing Streamlit..."
    python -m pip install --upgrade streamlit || {
        echo "❌ Failed to install Streamlit"
        echo "🔧 Trying with --user flag..."
        python -m pip install --user --upgrade streamlit
    }

    # Data processing
    echo "  📊 Installing data processing packages..."
    python -m pip install --upgrade --quiet \
        pandas \
        numpy \
        plotly \
        python-dotenv

    # Step 10: Install Snowflake dependencies
    echo "❄️ Installing Snowflake packages..."
    python -m pip install --upgrade --quiet \
        snowflake-connector-python

    # Step 11: Install additional utility packages
    echo "🛠️ Installing utility packages..."
    python -m pip install --upgrade --quiet \
        requests \
        pathlib \
        logging
fi

# Step 12: Verify critical files exist
echo "🔍 Verifying application files..."
REQUIRED_FILES=(
    "streamlit_app.py"
    "snowflake_bridge.py"
    "server.py"
)

OPTIONAL_FILES=(
    "test_framework.py"
    "config.json"
    "config_template.json"
    "run.py"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    else
        echo "✅ Found: $file"
    fi
done

# Check optional files
for file in "${OPTIONAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ Found (optional): $file"
    fi
done

if [ ${#MISSING_FILES[@]} -ne 0 ]; then
    echo "❌ Missing required files:"
    printf '   %s\n' "${MISSING_FILES[@]}"
    echo ""
    echo "📂 Current directory contents:"
    ls -la *.py 2>/dev/null || echo "No Python files found"
    echo ""
    echo "Please ensure all required Python files are in the same directory as this script."
    read -p "Press any key to exit..." -n1 -s
    exit 1
fi

# Step 13: Check configuration
echo "⚙️ Checking configuration..."
if [ -f "config.json" ]; then
    if grep -q "your-account" config.json 2>/dev/null; then
        echo "⚠️ Warning: Template values detected in config.json - using mock data mode"
        echo "💡 This is perfect for demos! Real Snowflake connection available when configured."
    else
        echo "✅ Custom Snowflake configuration detected"
    fi
else
    echo "ℹ️ No config.json found - application will use mock data (perfect for demos)"
fi

# Step 14: Test Python environment
echo "🧪 Testing Python environment..."
python -c "
try:
    import streamlit, pandas, numpy
    print('✅ Core packages imported successfully')
except ImportError as e:
    print(f'❌ Package import test failed: {e}')
    exit(1)
" || {
    echo "❌ Package import test failed"
    echo "🔧 There may be dependency issues"
    read -p "Press any key to exit..." -n1 -s
    exit 1
}

# Step 15: Set environment variables for better Streamlit experience
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Step 16: Pre-launch system check
echo "🔧 Running pre-launch system check..."
python -c "
import sys
print(f'Python version: {sys.version}')
print(f'Python executable: {sys.executable}')

try:
    from snowflake_bridge import ImprovedSnowflakeServer
    print('✅ Snowflake bridge module loaded successfully')
except Exception as e:
    print(f'⚠️ Snowflake bridge warning: {e}')

try:
    from server import SnowflakeMCPServer
    print('✅ Server compatibility layer loaded')
except Exception as e:
    print(f'⚠️ Server compatibility warning: {e}')
"

# Step 17: Launch the application
echo ""
echo "🚀 Launching Snowflake MCP Server with AI Assistant..."
echo "📱 Opening http://localhost:8501 in your browser..."
echo "📝 Logs will be written to: $LOG_FILE"
echo ""
echo "💡 Features available:"
echo "   • 🤖 AI-powered chat interface"
echo "   • 📊 Interactive data dashboard"
echo "   • 🔍 Advanced query tools"
echo "   • 📈 Data visualization and analysis"
echo "   • 💾 Data export capabilities"
echo "   • 🛠️ Admin and configuration tools"
echo ""
echo "🎯 Demo Mode: Application works immediately with mock data!"
echo "🔗 Real Snowflake: Configure config.json for live database access"
echo ""

# Start browser in background (delayed to allow server startup)
(sleep 5 && open "http://localhost:8501") &

# Launch Streamlit application with error handling
echo "🎯 Starting Streamlit server..."
python -m streamlit run streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --logger.level info 2>&1 | tee "$LOG_FILE"

# If streamlit command fails, try alternative
if [ $? -ne 0 ]; then
    echo "⚠️ Streamlit command failed, trying alternative launch method..."
    python streamlit_app.py 2>&1 | tee -a "$LOG_FILE"
fi

# Step 18: Post-execution cleanup and info
echo ""
echo "📊 Snowflake MCP Server has stopped."
echo ""

# Check for output files
if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR 2>/dev/null)" ]; then
    echo "📂 Generated files found in output directory:"
    ls -la "$OUTPUT_DIR"
    echo ""
    echo "🔍 Opening output directory..."
    open "$OUTPUT_DIR"
fi

# Check for downloaded data files
DATA_FILES=(*.csv *.json *.xlsx)
FOUND_DATA=false
for pattern in "${DATA_FILES[@]}"; do
    if ls $pattern 1> /dev/null 2>&1; then
        if [ "$FOUND_DATA" = false ]; then
            echo "📥 Downloaded data files found:"
            FOUND_DATA=true
        fi
        ls -la $pattern
    fi
done

if [ "$FOUND_DATA" = true ]; then
    echo ""
    echo "💡 Data files are ready for analysis!"
fi

# Show log file location
if [ -f "$LOG_FILE" ]; then
    echo "📋 Session log saved to: $LOG_FILE"
    echo "🔍 View logs: tail -f $LOG_FILE"
fi

echo ""
echo "✅ Snowflake MCP Server session completed."
echo "🔄 Run this script again anytime to restart the application."
echo ""
echo "💡 Tips for next time:"
echo "   • Use the AI Chat tab to explore your data naturally"
echo "   • Try the Query Tools for advanced SQL operations"
echo "   • Check the Dashboard for visual insights"
echo "   • Use Data Explorer to browse table structures"
echo "   • Configure real Snowflake connection in config.json"
echo ""

# Keep terminal open for user to see results
echo "👋 Thank you for using Snowflake MCP Server!"
echo ""
echo "Press any key to close this window..."
read -n 1 -s
