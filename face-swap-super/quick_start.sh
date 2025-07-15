#!/bin/bash

# Quick Start Script for Face Swap Super
# This script helps set up the Face Swap Super environment quickly

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🎭 Face Swap Super Quick Start${NC}"
echo "================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed. Please install Python 3.10 or higher.${NC}"
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print(sys.version_info[:2])")
if [[ "$python_version" < "(3, 10)" ]]; then
    echo -e "${RED}❌ Python 3.10 or higher is required. Current version: $python_version${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python version check passed${NC}"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}🔧 Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}⬆️ Upgrading pip...${NC}"
pip install --upgrade pip

# Install requirements
echo -e "${YELLOW}📥 Installing requirements...${NC}"
pip install -r requirements.txt

# Check if config file exists
if [ ! -f "config.yaml" ]; then
    echo -e "${YELLOW}⚙️ Creating default configuration...${NC}"
    python -c "from utils.config import create_default_config; create_default_config()"
fi

# Check for GPU support
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True; then
    echo -e "${GREEN}✅ CUDA GPU support detected${NC}"
else
    echo -e "${YELLOW}⚠️ No CUDA GPU detected. CPU mode will be used.${NC}"
fi

echo ""
echo -e "${GREEN}🎉 Setup complete!${NC}"
echo ""
echo "Quick start options:"
echo "1. Web interface:    python main.py --mode gradio"
echo "2. API server:       python main.py --mode server"
echo "3. Command line:     python main.py --mode cli --help"
echo "4. Docker:           docker-compose up"
echo ""
echo "For more information, see README.md"
echo ""
echo -e "${YELLOW}Note: On first run, models will be downloaded automatically (~2GB)${NC}"
