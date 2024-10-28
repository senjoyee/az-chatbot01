# build-and-deploy.ps1

# Function to wait for jobs and display their output
function Wait-JobsAndShowOutput {
    param (
        [Parameter(Mandatory=$true)]
        [System.Management.Automation.Job[]]$Jobs
    )
    
    Wait-Job -Job $Jobs | Out-Null
    foreach ($job in $Jobs) {
        Write-Host "`nOutput from $($job.Name):"
        # Filter out the PowerShell-specific messages
        Receive-Job -Job $job | Where-Object { 
            $_ -notmatch "CategoryInfo|FullyQualifiedErrorId|PSComputerName|RemoteException" 
        }
        Remove-Job -Job $job
    }
}

$currentDir = Get-Location

# Start parallel docker builds
$buildJobs = @(
    Start-Job -Name "BuildBackend" -ScriptBlock {
        param($workDir)
        Set-Location $workDir
        $output = docker build -t jssapmscr.azurecr.io/backend:latest ./backend 2>&1
        $output | Out-String
    } -ArgumentList $currentDir

    Start-Job -Name "BuildFrontend" -ScriptBlock {
        param($workDir)
        Set-Location $workDir
        $output = docker build -t jssapmscr.azurecr.io/frontend:latest ./frontend 2>&1
        $output | Out-String
    } -ArgumentList $currentDir
)

Write-Host "Starting Docker builds in parallel..."
Wait-JobsAndShowOutput -Jobs $buildJobs

# Login to Azure Container Registry
Write-Host "`nLogging into Azure Container Registry..."
az acr login --name jssapmscr

# Start parallel docker pushes
$pushJobs = @(
    Start-Job -Name "PushBackend" -ScriptBlock {
        $output = docker push jssapmscr.azurecr.io/backend:latest 2>&1
        $output | Out-String
    }
    Start-Job -Name "PushFrontend" -ScriptBlock {
        $output = docker push jssapmscr.azurecr.io/frontend:latest 2>&1
        $output | Out-String
    }
)

Write-Host "`nPushing Docker images in parallel..."
Wait-JobsAndShowOutput -Jobs $pushJobs

# Start parallel webapp restarts
$restartJobs = @(
    Start-Job -Name "RestartBackend" -ScriptBlock {
        $output = az webapp restart --name jscbbackend01 --resource-group js-sapservices-tools 2>&1
        $output | Out-String
    }
    Start-Job -Name "RestartDocumentChat" -ScriptBlock {
        $output = az webapp restart --name documentchatbot01 --resource-group js-sapservices-tools 2>&1
        $output | Out-String
    }
)

Write-Host "`nRestarting web apps in parallel..."
Wait-JobsAndShowOutput -Jobs $restartJobs

Write-Host "`nAll operations completed!"