# Stop-AzureResources.ps1

param (
    [string]$ResourceGroup = "js-sapservices-tools",
    [string]$FunctionAppName = "jsragfunc01",
    [string]$WebApp1 = "jscbbackend01",
    [string]$WebApp2 = "documentchatbot01",
    [string]$DbServer = "sapmscbpgdb"
)

Write-Host "Stopping Azure Resources..."

# Stop Function App
Write-Host "Stopping Function App: $FunctionAppName"
az functionapp stop --name $FunctionAppName --resource-group $ResourceGroup

# Stop Web Apps
Write-Host "Stopping Web Apps: $WebApp1, $WebApp2"
az webapp stop --name $WebApp1 --resource-group $ResourceGroup
az webapp stop --name $WebApp2 --resource-group $ResourceGroup

# Stop PostgreSQL Server
Write-Host "Stopping PostgreSQL Server: $DbServer"
az postgres flexible-server stop --resource-group $ResourceGroup --name $DbServer

Write-Host "All resources stopped successfully."