# Azure Web App Update Guide for Visual Studio Code on Windows

## 1. Introduction

This guide outlines the process for updating a multi-container Azure Web App using Visual Studio Code (VS Code) on Windows. It covers updating source code, building and pushing new container images, and deploying updates to Azure Web Apps.

## 2. Prerequisites

- Visual Studio Code installed on Windows
- Docker Desktop for Windows
- Azure CLI
- Azure Account extension for VS Code
- Docker extension for VS Code
- Azure App Service extension for VS Code

## 3. Setting Up Your Development Environment

1. Install Visual Studio Code from https://code.visualstudio.com/
2. Install Docker Desktop from https://www.docker.com/products/docker-desktop
3. Install Azure CLI:
   ```
   winget install -e --id Microsoft.AzureCLI
   ```
4. Install required VS Code extensions:
   - Azure Account
   - Docker
   - Azure App Service

## 4. Updating Your Source Code with VS Code

1. Open your project folder in VS Code:
   ```
   code C:\path\to\your\project
   ```
2. Make necessary changes to your source code
3. Save your changes (Ctrl + S)

## 5. Building and Pushing New Container Images

### Using Docker extension in VS Code:

1. Right-click on your Dockerfile
2. Select "Build Image..."
3. Enter a tag for your image (e.g., `myregistry.azurecr.io/myapp:v1.0.1`)
4. Right-click on the new image in the Docker extension
5. Select "Push..."

### Using command line in VS Code terminal:

1. Open a new terminal in VS Code (Ctrl + `)
2. Build your Docker image:
   ```
   docker build -t myregistry.azurecr.io/myapp:v1.0.1 .
   ```
3. Push the image to Azure Container Registry:
   ```
   docker push myregistry.azurecr.io/myapp:v1.0.1
   ```

## 6. Updating Azure Web Apps

### Using Azure App Service extension in VS Code:

1. Click on the Azure icon in the VS Code sidebar
2. Expand your subscription and right-click on your Web App
3. Select "Deploy to Web App..."
4. Choose the image you just pushed

### Using Azure CLI in VS Code terminal:

1. Log in to Azure:
   ```
   az login
   ```
2. Update your Web App with the new image:
   ```
   az webapp config container set --name mywebapp --resource-group myresourcegroup --docker-custom-image-name myregistry.azurecr.io/myapp:v1.0.1
   ```

## 7. Setting Up Continuous Integration/Continuous Deployment (CI/CD)

### Azure Pipelines integration with VS Code:

1. Install the Azure Pipelines extension for VS Code
2. Press Ctrl + Shift + P and type "Azure Pipelines: Create Pipeline"
3. Follow the wizard to create a new pipeline
4. Example YAML for a basic pipeline:

```yaml
trigger:
- main

pool:
  vmImage: 'windows-latest'

steps:
- task: Docker@2
  inputs:
    containerRegistry: 'myAzureContainerRegistry'
    repository: 'myapp'
    command: 'buildAndPush'
    Dockerfile: '**/Dockerfile'
    tags: '$(Build.BuildId)'

- task: AzureWebAppContainer@1
  inputs:
    azureSubscription: 'myAzureSubscription'
    appName: 'mywebapp'
    imageName: 'myregistry.azurecr.io/myapp:$(Build.BuildId)'
```

## 8. Troubleshooting

- If you encounter Docker-related issues, ensure Docker Desktop is running
- For Azure CLI errors, try logging out and logging in again:
  ```
  az logout
  az login
  ```
- Check your Azure subscription is set correctly:
  ```
  az account show
  az account set --subscription <subscription-id>
  ```

## 9. Best Practices for Windows Development with VS Code

- Use Windows Subsystem for Linux (WSL) for a Linux-like environment
- Keep your Docker images and VS Code extensions up to date
- Use version control (e.g., Git) and commit changes frequently
- Utilize VS Code's integrated terminal for command-line operations
- Leverage VS Code's task system to automate repetitive tasks