#!/usr/bin/env pwsh

# Stop on first error
$ErrorActionPreference = "Stop"

# Function to show step status
function Write-StepStatus {
    param(
        [string]$step,
        [string]$status,
        [string]$message
    )
    $color = if ($status -eq "SUCCESS") { "Green" } else { "Red" }
    Write-Host "`n[$step] $status" -ForegroundColor $color
    if ($message) {
        Write-Host $message
    }
}

try {
    # Step 1: Build Docker image
    Write-Host "`nStep 1: Building Docker image..." -ForegroundColor Cyan
    docker build -t jssapmscr.azurecr.io/frontend:latest ./frontend
    Write-StepStatus "Build" "SUCCESS" "Docker image built successfully"

    # Step 2: Login to Azure Container Registry
    Write-Host "`nStep 2: Logging into Azure Container Registry..." -ForegroundColor Cyan
    az acr login --name jssapmscr
    Write-StepStatus "Login" "SUCCESS" "Successfully logged into ACR"

    # Step 3: Push Docker image
    Write-Host "`nStep 3: Pushing Docker image to ACR..." -ForegroundColor Cyan
    docker push jssapmscr.azurecr.io/frontend:latest
    Write-StepStatus "Push" "SUCCESS" "Docker image pushed successfully"

    Write-Host "`nAll steps completed successfully!" -ForegroundColor Green
}
catch {
    Write-StepStatus "Error" "FAILED" $_.Exception.Message
    exit 1
}
