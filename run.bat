@echo off
REM PDF Heading Extractor Docker Runner Script for Windows

if "%1"=="" goto usage

if "%1"=="build" goto build
if "%1"=="start" goto start
if "%1"=="stop" goto stop
if "%1"=="restart" goto restart
if "%1"=="logs" goto logs
if "%1"=="shell" goto shell
if "%1"=="process" goto process
if "%1"=="prepare" goto prepare
if "%1"=="train" goto train
if "%1"=="predict" goto predict
if "%1"=="help" goto usage
if "%1"=="--help" goto usage
if "%1"=="-h" goto usage

echo Unknown command: %1
echo.
goto usage

:build
echo Building Docker image...
docker-compose build
goto end

:start
echo Starting container...
docker-compose up -d
goto end

:stop
echo Stopping container...
docker-compose down
goto end

:restart
echo Restarting container...
docker-compose down
docker-compose up -d
goto end

:logs
docker-compose logs -f
goto end

:shell
docker exec -it pdf-heading-extractor /bin/bash
goto end

:process
docker exec -it pdf-heading-extractor python process_pdfs.py
goto end

:prepare
docker exec -it pdf-heading-extractor python ml/prepare_labeling_csv.py
goto end

:train
docker exec -it pdf-heading-extractor python ml/train_model.py
goto end

:predict
docker exec -it pdf-heading-extractor python ml/predict_headings.py
goto end

:usage
echo PDF Heading Extractor Docker Runner
echo.
echo Usage: run.bat [COMMAND]
echo.
echo Commands:
echo   build     - Build the Docker image
echo   start     - Start the container
echo   stop      - Stop the container
echo   restart   - Restart the container
echo   logs      - Show container logs
echo   shell     - Open shell in running container
echo   process   - Run PDF processing
echo   prepare   - Run data preparation
echo   train     - Train the model
echo   predict   - Run predictions on new PDFs
echo   help      - Show this help message
echo.
echo Examples:
echo   run.bat build
echo   run.bat process
echo   run.bat shell
echo.
goto end

:end 