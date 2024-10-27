# Define each job as a Start-Job command
$job1 = Start-Job -ScriptBlock { docker build -t jssapmscr.azurecr.io/backend:latest ./backend }
$job2 = Start-Job -ScriptBlock { docker build -t jssapmscr.azurecr.io/frontend:latest ./frontend }
$job3 = Start-Job -ScriptBlock { docker build -t jssapmscr.azurecr.io/proxy:latest ./proxy }

# Start additional tasks
$job4 = Start-Job -ScriptBlock { az acr login --name jssapmscr }
$job5 = Start-Job -ScriptBlock { docker push jssapmscr.azurecr.io/backend:latest }
$job6 = Start-Job -ScriptBlock { docker push jssapmscr.azurecr.io/frontend:latest }
$job7 = Start-Job -ScriptBlock { docker push jssapmscr.azurecr.io/proxy:latest }

# Restarting web apps in parallel
$job8 = Start-Job -ScriptBlock { az webapp restart --name jscbbackend01 --resource-group js-sapservices-tools }
$job9 = Start-Job -ScriptBlock { az webapp restart --name documentchatbot01 --resource-group js-sapservices-tools }
$job10 = Start-Job -ScriptBlock { az webapp restart --name jscb-proxy-nginx --resource-group js-sapservices-tools }

# Wait for all jobs to complete
Get-Job | Wait-Job

# Receive job output if needed (optional)
$results = Get-Job | ForEach-Object { Receive-Job -Job $_ }
$results
