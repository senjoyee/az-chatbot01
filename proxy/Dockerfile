# Use the official nginx image as a parent image
FROM nginx:alpine

# Remove the default nginx configuration file
RUN rm /etc/nginx/conf.d/default.conf

# Copy the custom nginx configuration file
COPY nginx.conf /etc/nginx/conf.d/

# Expose port 80
EXPOSE 80

# Start nginx when the container launches
CMD ["nginx", "-g", "daemon off;"]