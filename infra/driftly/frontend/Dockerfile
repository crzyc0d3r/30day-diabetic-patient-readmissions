# Driftly web — multi-stage: build the Vite bundle with Node, serve it with nginx.
# Build context is infra/driftly/frontend (see infra/docker-compose.yml).
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
# Use a clean, reproducible install when a lockfile is present; fall back to a
# plain install on first build before a lockfile is committed.
RUN npm ci || npm install
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=4 \
    CMD curl -fsS http://localhost/ -o /dev/null || exit 1
