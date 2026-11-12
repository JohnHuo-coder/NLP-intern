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

- RDS: `db.t3.micro`, 30GB, single-AZ, short backups, no NAT.
- Fargate: always-on task (2 vCPU / 4 GB) + ALB hours + public IPv4 — often similar or higher than RDS for a demo.
- Stop/destroy stacks when not demoing.

## 8) GitHub Actions CI + manual CD

CI (`.github/workflows/ci.yml`) runs on push/PR to `main`:

- `pytest tests/test_text_cleaning_and_query_parse.py tests/test_semantic_text_llm.py`
- `cdk synth` of the RDS + Fargate templates (no deploy)

CD (`.github/workflows/deploy.yml`) is **manual only**: GitHub → Actions → **Deploy Fargate** → **Run workflow** → `deploy-fargate` or `destroy-fargate`. Merge to `main` does **not** deploy (Fargate is expensive when idle).

The deploy job uses `--exclusively` on `RealEstateFargateApiStack`. It does **not** update RDS, so console changes (private access, security group) stay put.

### One-time AWS: GitHub OIDC

From `infra/cdk` (venv on, AWS creds for account `212541130670`):

```powershell
cdk deploy GitHubActionsOidcStack `
  -c account=212541130670 `
  -c region=us-east-2 `
  -c enableGithubOidc=true `
  -c githubOwner=JohnHuo-coder `
  -c githubRepo=NLP-intern
```

Copy output `RoleArn`. Trust the role in the existing CDK bootstrap:

```powershell
cdk bootstrap aws://212541130670/us-east-2 `
  --trust <RoleArn> `
  --cloudformation-execution-policies arn:aws:iam::aws:policy/AdministratorAccess
```

`--trust` lets this GitHub role assume the bootstrap `cdk-hnb659fds-*` roles. The CloudFormation execution policy is what **CloudFormation** may create, not a permission on the GitHub role itself.

If the account already has `token.actions.githubusercontent.com` as an OIDC provider, add `-c githubOidcExistingProvider=true`.

### GitHub Environment `aws-us-east-2`

Create environment **aws-us-east-2** (optional required reviewers). Variables:

- `CDK_ACCOUNT` = `212541130670`
- `AWS_REGION` = `us-east-2`
- `S3_BUCKET` = `real-estate-search-212541130670-us-east-2-an`
- `S3_PREFIX` = empty (objects are at the bucket root)
- `OPENAI_SECRET_ARN` = `arn:aws:secretsmanager:us-east-2:212541130670:secret:openai-api-key-SJFp3Y`

Secret:

- `AWS_ROLE_ARN` = `RoleArn` from `GitHubActionsOidcStack`

Do not put the OpenAI key or AWS access keys in GitHub. After deploy, the workflow prints `ApiUrl` (a new ALB DNS if the stack was recreated).
