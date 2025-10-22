# React 프론트엔드를 위한 멀티 스테이지 Dockerfile

# 개발 단계
FROM node:18-alpine as development

WORKDIR /app

# 패키지 파일 복사
COPY package*.json ./

# 의존성 설치
RUN npm ci --only=production=false

# 소스 코드 복사
COPY . .

# 개발 서버 실행
EXPOSE 3000
CMD ["npm", "start"]

# 빌드 단계
FROM node:18-alpine as build

WORKDIR /app

# 패키지 파일 복사 및 설치
COPY package*.json ./
RUN npm ci --only=production=false

# 소스 코드 복사
COPY . .

# React 앱 빌드
RUN npm run build

# 프로덕션 단계
FROM nginx:alpine as production

# 빌드된 파일을 nginx에 복사
COPY --from=build /app/build /usr/share/nginx/html

# nginx 설정 파일 복사
COPY nginx.conf /etc/nginx/conf.d/default.conf

# nginx 포트 노출
EXPOSE 80

# nginx 실행
CMD ["nginx", "-g", "daemon off;"]