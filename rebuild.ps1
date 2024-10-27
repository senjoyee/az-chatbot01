docker build -t jssapmscr.azurecr.io/backend:latest ./backend
docker build -t jssapmscr.azurecr.io/frontend:latest ./frontend
docker build -t jssapmscr.azurecr.io/proxy:latest ./proxy
az acr login --name jssapmscr
docker push jssapmscr.azurecr.io/backend:latest
docker push jssapmscr.azurecr.io/frontend:latest
docker push jssapmscr.azurecr.io/proxy:latest
az webapp restart --name jscbbackend01 --resource-group js-sapservices-tools
az webapp restart --name documentchatbot01 --resource-group js-sapservices-tools
az webapp restart --name jscb-proxy-nginx --resource-group js-sapservices-tools