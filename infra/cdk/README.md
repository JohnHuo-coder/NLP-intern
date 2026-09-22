# AWS CDK - RDS + Fargate (MVP)

This CDK app can create:

1. **RDS stack** (`RealEstateRdsMvpStack`)
   - VPC (2 AZ, no NAT to reduce cost)
   - Security Group for MySQL
   - Amazon RDS MySQL 8.0 (`db.t3.small`)
   - Secrets Manager credentials

2. **Fargate stack** (`RealEstateFargateApiStack`, optional)
   - ECS Cluster + Fargate service
   - Application Load Balancer (public HTTP)
   - Docker image built from repo root `Dockerfile` and pushed via CDK assets/ECR
   - Task env/secrets wired to RDS + S3 artifacts bucket

## 1) Prerequisites

- AWS credentials (`aws login` / `aws configure`)
- Node.js + `aws-cdk` CLI (script installs globally)
- For Fargate deploy: **Docker Desktop running** (CDK builds the image locally)
- Processed artifacts already uploaded to S3 (see app `S3_BUCKET` / `S3_PREFIX`)

## 2) Deploy RDS only

```powershell
.\infra\cdk\deploy.ps1 `
  -StackName RealEstateRdsMvpStack `
  -Region us-east-1 `
  -DbName real_estate `
  -DbUsername admin `
  -DbAccessCidr 0.0.0.0/0 `
  -PublicAccess true
```

## 3) Deploy RDS + Fargate API

```powershell
.\infra\cdk\deploy.ps1 `
  -DeployFargate `
  -S3Bucket ealestate-search-dev-313117918514-us-east-1-an `
  -S3Prefix data/processed `
  -Region us-east-1 `
  -DbAccessCidr 0.0.0.0/0 `
  -PublicAccess true
```

Defaults: **2 vCPU / 4 GB** memory, desired count `1`. Override with `-FargateCpu`, `-FargateMemory`, `-FargateDesiredCount`.

After deploy, stack output `ApiUrl` is the public ALB base URL. Try:

- `GET {ApiUrl}/health`
- `GET {ApiUrl}/ready`
- `POST {ApiUrl}/api/query`

## 4) Local Docker smoke test (before Fargate)

From repo root (Docker running):

```powershell
docker build -t real-estate-api .
docker run --rm -p 8000:8000 `
  -e MYSQLHOST=... `
  -e MYSQLPORT=3306 `
  -e MYSQLUSER=... `
  -e MYSQLPASSWORD=... `
  -e MYSQLDATABASE=real_estate `
  -e S3_BUCKET=ealestate-search-dev-313117918514-us-east-1-an `
  -e S3_PREFIX=data/processed `
  -e AWS_REGION=us-east-1 `
  -e AWS_ACCESS_KEY_ID=... `
  -e AWS_SECRET_ACCESS_KEY=... `
  real-estate-api
```

Image does **not** bake `data/processed`; the app downloads from S3 at startup (same as Fargate).

## 5) Networking notes (cost-aware MVP)

- VPC has **no NAT gateway**.
- Fargate tasks run in **public subnets with public IPs** so they can pull from ECR and reach S3/Secrets Manager.
- Task security group is allowed into the RDS security group on port `3306`.
- ALB health check uses `/health` (not `/ready`) so load balancer does not fail during long startup.

## 6) Destroy

```powershell
cd .\infra\cdk
.\.venv\Scripts\Activate.ps1
# Destroy API first if Fargate stack exists
cdk destroy RealEstateFargateApiStack
cdk destroy RealEstateRdsMvpStack
```

## 7) Cost notes

- RDS: `db.t3.small`, 30GB, single-AZ, short backups, no NAT.
- Fargate: always-on task (2 vCPU / 4 GB) + ALB hours + public IPv4 — often similar or higher than RDS for a demo.
- Stop/destroy stacks when not demoing.
