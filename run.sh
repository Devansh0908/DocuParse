#!/bin/bash

# PDF Heading Extractor Docker Runner Script

# Function to show usage
show_usage() {
    echo "PDF Heading Extractor Docker Runner"
    echo ""
    echo "Usage: ./run.sh [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build     - Build the Docker image"
    echo "  start     - Start the container"
    echo "  stop      - Stop the container"
    echo "  restart   - Restart the container"
    echo "  logs      - Show container logs"
    echo "  shell     - Open shell in running container"
    echo "  process   - Run PDF processing"
    echo "  prepare   - Run data preparation"
    echo "  train     - Train the model"
    echo "  predict   - Run predictions on new PDFs"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh build"
    echo "  ./run.sh process"
    echo "  ./run.sh shell"
}

# Build the Docker image
build_image() {
    echo "Building Docker image..."
    docker-compose build
}

# Start the container
start_container() {
    echo "Starting container..."
    docker-compose up -d
}

# Stop the container
stop_container() {
    echo "Stopping container..."
    docker-compose down
}

# Show logs
show_logs() {
    docker-compose logs -f
}

# Open shell in container
open_shell() {
    docker exec -it pdf-heading-extractor /bin/bash
}

# Run PDF processing
run_process() {
    docker exec -it pdf-heading-extractor python process_pdfs.py
}

# Run data preparation
run_prepare() {
    docker exec -it pdf-heading-extractor python ml/prepare_labeling_csv.py
}

# Run model training
run_train() {
    docker exec -it pdf-heading-extractor python ml/train_model.py
}

# Run predictions
run_predict() {
    docker exec -it pdf-heading-extractor python ml/predict_headings.py
}

# Main script logic
case "$1" in
    build)
        build_image
        ;;
    start)
        start_container
        ;;
    stop)
        stop_container
        ;;
    restart)
        stop_container
        start_container
        ;;
    logs)
        show_logs
        ;;
    shell)
        open_shell
        ;;
    process)
        run_process
        ;;
    prepare)
        run_prepare
        ;;
    train)
        run_train
        ;;
    predict)
        run_predict
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo "Unknown command: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac 