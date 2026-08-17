# syntax=docker/dockerfile:1
FROM node:22-bookworm-slim AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM node:22-bookworm-slim

WORKDIR /app

ENV API_PROXY_TARGET=http://backend:8000

COPY --from=build /app /app

EXPOSE 5173

HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=8 \
    CMD node -e "fetch('http://127.0.0.1:5173/').then((r)=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0", "--port", "5173"]
