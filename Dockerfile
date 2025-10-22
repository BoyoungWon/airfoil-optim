# X-foil 최적화 시스템을 위한 Docker 환경
FROM ubuntu:22.04

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    curl \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉터리 생성
WORKDIR /app

# X-foil 소스코드 다운로드 및 컴파일
RUN wget https://web.mit.edu/drela/Public/web/xfoil/xfoil6.99.tgz && \
    tar -xzf xfoil6.99.tgz && \
    cd xfoil6.99 && \
    cd src && \
    make install

# X-foil 실행파일을 PATH에 추가
ENV PATH="/app/xfoil6.99/bin:${PATH}"

# Python 가상환경 생성 및 필요 패키지 설치
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY . .

# 포트 노출
EXPOSE 8000 3000

# 시작 스크립트 실행
CMD ["./start.sh"]