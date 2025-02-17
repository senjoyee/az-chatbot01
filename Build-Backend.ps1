# Build-Backend.ps1

# Set error action preference to stop on any error
$ErrorActionPreference = "Stop"

Write-Host "Starting Docker build process..." -ForegroundColor Green

try {
    # Login to Azure Container Registry first
    Write-Host "Logging into Azure Container Registry..." -ForegroundColor Green
    az acr login --name jssapmscr
    if ($LASTEXITCODE -ne 0) {
        throw "ACR login failed"
    }
    Write-Host "ACR login successful" -ForegroundColor Green

    # Pull the latest image for cache
    Write-Host "Pulling latest image for cache..." -ForegroundColor Green
    docker pull jssapmscr.azurecr.io/backend:latest
    # Don't throw on pull failure as the image might not exist yet
    Write-Host "Pull completed" -ForegroundColor Green

    # Build the Docker image with cache options
    Write-Host "Building Docker image..." -ForegroundColor Green
    docker build --cache-from jssapmscr.azurecr.io/backend:latest -t jssapmscr.azurecr.io/backend:latest ./backend
    if ($LASTEXITCODE -ne 0) {
        throw "Docker build failed"
    }
    Write-Host "Docker build completed successfully" -ForegroundColor Green

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
