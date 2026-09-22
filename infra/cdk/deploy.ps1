param(
  [string]$StackName = "RealEstateRdsMvpStack",
  [string]$FargateStackName = "RealEstateFargateApiStack",
  [string]$Region = "us-east-1",
  [string]$DbName = "real_estate",
  [string]$DbUsername = "admin",
  [string]$DbAccessCidr = "0.0.0.0/0",
  [string]$PublicAccess = "true",
  [switch]$DeployFargate,
  [string]$S3Bucket = "",
  [string]$S3Prefix = "data/processed",
  [string]$CorsAllowedOrigins = "*",
  [int]$FargateCpu = 2048,
  [int]$FargateMemory = 4096,
  [int]$FargateDesiredCount = 1
)

$ErrorActionPreference = "Stop"

Write-Host "==> Resolving AWS account..."
$account = aws sts get-caller-identity --query Account --output text
if (-not $account) {
  throw "Unable to resolve AWS account. Run 'aws configure' or SSO login first."
}

if ($DeployFargate -and -not $S3Bucket) {
  throw "DeployFargate requires -S3Bucket <bucket-name> (processed artifacts)."
}

Write-Host "==> Entering infra/cdk..."
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
  Write-Host "==> Creating venv..."
  py -m venv .venv
}

Write-Host "==> Activating venv..."
& ".\.venv\Scripts\Activate.ps1"

Write-Host "==> Installing dependencies..."
pip install -r requirements.txt
npm install -g aws-cdk

$env:CDK_DEFAULT_ACCOUNT = $account
$env:CDK_DEFAULT_REGION = $Region

Write-Host "==> Bootstrapping CDK (safe to re-run)..."
cdk bootstrap "aws://$account/$Region"

$enableFargate = if ($DeployFargate) { "true" } else { "false" }
$contextArgs = @(
  "-c", "account=$account",
  "-c", "region=$Region",
  "-c", "stackName=$StackName",
  "-c", "fargateStackName=$FargateStackName",
  "-c", "dbName=$DbName",
  "-c", "dbUsername=$DbUsername",
  "-c", "dbAccessCidr=$DbAccessCidr",
  "-c", "publicAccess=$PublicAccess",
  "-c", "enableFargate=$enableFargate",
  "-c", "s3Bucket=$S3Bucket",
  "-c", "s3Prefix=$S3Prefix",
  "-c", "corsAllowedOrigins=$CorsAllowedOrigins",
  "-c", "fargateCpu=$FargateCpu",
  "-c", "fargateMemory=$FargateMemory",
  "-c", "fargateDesiredCount=$FargateDesiredCount"
)

if ($DeployFargate) {
  Write-Host "==> Deploying RDS + Fargate stacks (Docker required for image build)..."
  cdk deploy $StackName $FargateStackName `
    --require-approval never `
    @contextArgs
} else {
  Write-Host "==> Deploying RDS stack $StackName ..."
  cdk deploy $StackName `
    --require-approval never `
    @contextArgs
}

Write-Host "==> Done."
