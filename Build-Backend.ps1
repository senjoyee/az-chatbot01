# build-and-push.ps1

# Set error action preference to stop on any error
$ErrorActionPreference = "Stop"

Write-Host "Starting Docker build process..." -ForegroundColor Green
try {
    # Build the Docker image
    docker build -t jssapmscr.azurecr.io/backend:latest ./backend
    if ($LASTEXITCODE -ne 0) {
        throw "Docker build failed"
    }
    Write-Host "Docker build completed successfully" -ForegroundColor Green

    # Login to Azure Container Registry
    Write-Host "Logging into Azure Container Registry..." -ForegroundColor Green
    az acr login --name jssapmscr
    if ($LASTEXITCODE -ne 0) {
        throw "ACR login failed"
    }
    Write-Host "ACR login successful" -ForegroundColor Green

    # Push the Docker image
    Write-Host "Pushing Docker image to ACR..." -ForegroundColor Green
    docker push jssapmscr.azurecr.io/backend:latest
    if ($LASTEXITCODE -ne 0) {
        throw "Docker push failed"
    }
    Write-Host "Docker image pushed successfully" -ForegroundColor Green
} catch {
    Write-Host "An error occurred: $_" -ForegroundColor Red
    exit 1
}

Write-Host "Process completed successfully!" -ForegroundColor Green

# Created/Modified files during execution:
# build-and-push.ps1