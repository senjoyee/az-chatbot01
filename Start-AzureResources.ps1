# Start-AzureResources.ps1

param (
    [string]$ResourceGroup = "js-sapservices-tools",
    [string]$FunctionAppName = "jsragfunc01",
    [string]$WebApp1 = "jscbbackend01",
    [string]$WebApp2 = "documentchatbot01",
    [string]$DbServer = "sapmscbpgdb"
)

Write-Host "Starting Azure Resources..."

# Start PostgreSQL Server (start DB first)
Write-Host "Starting PostgreSQL Server: $DbServer"
az postgres flexible-server start --resource-group $ResourceGroup --name $DbServer

# Wait for DB to be ready
Start-Sleep -Seconds 30

# Start Web Apps
Write-Host "Starting Web Apps: $WebApp1, $WebApp2"
az webapp start --name $WebApp1 --resource-group $ResourceGroup
az webapp start --name $WebApp2 --resource-group $ResourceGroup

# Start Function App
Write-Host "Starting Function App: $FunctionAppName"
az functionapp start --name $FunctionAppName --resource-group $ResourceGroup

Write-Host "All resources started successfully."