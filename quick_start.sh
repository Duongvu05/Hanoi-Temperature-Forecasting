#!/bin/bash

# Hanoi Temperature Forecasting - Quick Start Script
# This script sets up the environment and demonstrates key functionality

echo "🌡️ Hanoi Temperature Forecasting - Quick Start"
echo "================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 found"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚙️ Setting up environment configuration..."
    cp .env.example .env
    echo "📝 Please edit .env file and add your Visual Crossing API key"
    echo "   You can get a free API key from: https://www.visualcrossing.com/"
    echo ""
fi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/raw/daily data/raw/hourly
mkdir -p data/processed/daily data/processed/hourly
mkdir -p models/daily models/hourly models/onnx
mkdir -p logs

echo ""
echo "🚀 Setup completed! Here's what you can do next:"
echo ""
echo "1. 📊 Start the Streamlit UI:"
echo "   streamlit run ui/app.py"
echo ""
echo "2. 📓 Explore the Jupyter notebooks:"
echo "   jupyter notebook notebooks/"
echo ""
echo "3. 🌐 Collect weather data (requires API key):"
echo "   python src/data/collector.py --data-type daily --years 10"
echo ""
echo "4. 🔍 Process collected data:"
echo "   python src/data/processor.py data/raw/daily/your_data.csv"
echo ""
echo "5. 📖 Read the documentation:"
echo "   Open README.md in your preferred editor"
echo ""

# Check if API key is set (basic check)
if [ -f ".env" ]; then
    if grep -q "your_visual_crossing_api_key_here" .env; then
        echo "⚠️  IMPORTANT: Don't forget to set your Visual Crossing API key in .env file!"
    fi
fi

echo ""
echo "💡 For detailed instructions, please refer to the README.md file"
echo "🎯 Happy forecasting! 🌤️"