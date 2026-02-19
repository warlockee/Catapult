# Build stage
FROM node:22-slim as builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN yarn install

# Copy source code
COPY . .

# Build arguments
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL
ARG VITE_DEFAULT_API_KEY
ENV VITE_DEFAULT_API_KEY=$VITE_DEFAULT_API_KEY

# Build application
RUN yarn build

# Production stage (optional - nginx can serve directly from builder)
FROM nginx:alpine as production

# Copy built files from builder
COPY --from=builder /app/build /usr/share/nginx/html

# Fix permissions for all files
RUN find /usr/share/nginx/html -type f -exec chmod 644 {} \; && \
    find /usr/share/nginx/html -type d -exec chmod 755 {} \;

# Default nginx config (will be overridden by docker-compose volume mount)
RUN rm /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
