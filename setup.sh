#!/bin/bash
# Setup script for Market Intelligence MVP

set -e  # Exit on error

echo "🚀 Market Intelligence MVP - Setup Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "ℹ $1"
}

# Check if Docker is installed
echo "Checking prerequisites..."
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi
print_success "Docker is installed"

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi
print_success "Docker Compose is installed"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    print_warning "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    print_success "uv installed successfully"
else
    print_success "uv is already installed"
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    print_info "Creating .env file from template..."
    cp .env.example .env
    print_warning "Please edit .env file and add your API keys!"
    echo ""
    echo "Required API keys:"
    echo "  1. ALPHA_VANTAGE_API_KEY - Get from: https://www.alphavantage.co/support/#api-key"
    echo "  2. NEWS_API_KEY - Get from: https://newsapi.org/register"
    echo ""
else
    print_success ".env file already exists"
fi

# Create necessary directories
print_info "Creating project directories..."
mkdir -p data/{raw,processed,features,external}
mkdir -p models
mkdir -p airflow/{logs,plugins,config}
mkdir -p notebooks
mkdir -p benchmarks/results
print_success "Directories created"

# Create Python virtual environment
print_info "Creating Python virtual environment..."
if [ ! -d ".venv" ]; then
    uv venv
    print_success "Virtual environment created"
else
    print_success "Virtual environment already exists"
fi

# Activate virtual environment and install dependencies
print_info "Installing Python dependencies..."
source .venv/bin/activate || . .venv/Scripts/activate
uv pip install -e .
print_success "Dependencies installed"

# Set proper permissions for Airflow
print_info "Setting Airflow permissions..."
export AIRFLOW_UID=$(id -u)
echo "AIRFLOW_UID=${AIRFLOW_UID}" >> .env
print_success "Airflow UID set to ${AIRFLOW_UID}"

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    print_info "Initializing Git repository..."
    git init
    git add .
    git commit -m "Initial commit: Project setup"
    print_success "Git repository initialized"
else
    print_success "Git repository already exists"
fi

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Edit .env file and add your API keys:"
echo "   nano .env"
echo ""
echo "2. Start Docker services:"
echo "   docker-compose up -d"
echo ""
echo "3. Access the services:"
echo "   - Airflow UI: http://localhost:8080 (airflow/airflow)"
echo "   - MLflow UI: http://localhost:5000"
echo "   - Jupyter Lab: http://localhost:8888 (token: jupyter)"
echo ""
echo "4. Trigger the data ingestion pipeline:"
echo "   docker-compose exec airflow-scheduler airflow dags trigger data_ingestion_pipeline"
echo ""
echo "5. View the project roadmap:"
echo "   cat warp.md"
echo ""
echo "Happy coding! 🎉"
