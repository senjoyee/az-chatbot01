# Get the current directory where the script is running
$currentPath = Get-Location

# Function to wait for jobs and display status
function Wait-JobsWithStatus {
    param(
        [string]$stage,
        [switch]$suppressOutput = $false
    )
    Write-Host "Waiting for $stage to complete..." -ForegroundColor Cyan
    
    $jobs = Get-Job
    $hasErrors = $false
    
    foreach ($job in $jobs) {
        $result = Receive-Job -Job $job -Wait
        if ($job.State -eq 'Failed') {
            $hasErrors = $true
            Write-Host "Error in $stage job:" -ForegroundColor Red
            Write-Host $result
        }
        elseif (-not $suppressOutput) {
            # For docker builds, only show actual errors
            if ($stage -eq "docker builds") {
                if ($result -match "error|failed" -and $result -notmatch "NotSpecified") {
                    Write-Host $result -ForegroundColor Red
                    $hasErrors = $true
                }
            }
            else {
                Write-Host $result
            }
        }
    }
    
    Get-Job | Remove-Job
    
    if ($hasErrors) {
        Write-Host "$stage had failures. Stopping script." -ForegroundColor Red
        exit 1
    }
    else {
        Write-Host "$stage completed successfully." -ForegroundColor Green
    }
}

# Verify directories exist before proceeding
$directories = @("backend", "frontend", "proxy")
foreach ($dir in $directories) {
    $fullPath = Join-Path $currentPath $dir
    if (-not (Test-Path $fullPath)) {
        Write-Host "Directory not found: $fullPath" -ForegroundColor Red
        exit 1
    }
}

# Step 1: Run docker builds in parallel
Write-Host "Starting docker builds..." -ForegroundColor Cyan
$builds = @(
    @{Name = "backend"; Path = "./backend"},
    @{Name = "frontend"; Path = "./frontend"},
    @{Name = "proxy"; Path = "./proxy"}
)

foreach ($build in $builds) {
    Start-Job -ScriptBlock { 
        param($path, $currentPath)
        Set-Location $currentPath
        $output = docker build -t "jssapmscr.azurecr.io/$($path.Substring(2)):latest" $path 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw $output
        }
        return $output
    } -ArgumentList $build.Path, $currentPath
}

# Wait for builds to complete
Wait-JobsWithStatus "docker builds" -suppressOutput

# Step 2: Azure Container Registry login
Write-Host "Logging into Azure Container Registry..." -ForegroundColor Cyan
$loginResult = az acr login --name jssapmscr 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to login to Azure Container Registry" -ForegroundColor Red
    Write-Host $loginResult
    exit 1
}
Write-Host "Successfully logged into Azure Container Registry" -ForegroundColor Green

# Step 3: Run docker pushes in parallel
Write-Host "Starting docker pushes..." -ForegroundColor Cyan
foreach ($build in $builds) {
    $imageName = "jssapmscr.azurecr.io/$($build.Name):latest"
    Start-Job -ScriptBlock { 
        param($image)
        $output = docker push $image 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw $output
        }
        return $output
    } -ArgumentList $imageName
}

# Wait for pushes to complete
Wait-JobsWithStatus "docker pushes"

# Step 4: Run webapp restarts in parallel
Write-Host "Starting webapp restarts..." -ForegroundColor Cyan
$webapps = @(
    @{Name = "jscbbackend01"},
    @{Name = "documentchatbot01"},
    @{Name = "jscb-proxy-nginx"}
)

foreach ($webapp in $webapps) {
    Start-Job -ScriptBlock { 
        param($name)
        $output = az webapp restart --name $name --resource-group js-sapservices-tools 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw $output
        }
        return $output
    } -ArgumentList $webapp.Name
}

# Wait for restarts to complete
Wait-JobsWithStatus "webapp restarts"

Write-Host "All operations completed successfully!" -ForegroundColor Green