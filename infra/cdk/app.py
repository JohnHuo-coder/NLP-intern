#!/usr/bin/env python3
import os
from pathlib import Path

import aws_cdk as cdk

from infra_cdk.fargate_stack import FargateApiStack
from infra_cdk.rds_stack import RdsMvpStack


def context_or_env(app: cdk.App, key: str, env_key: str, default: str) -> str:
    value = app.node.try_get_context(key)
    if value is not None:
        return str(value)
    return os.getenv(env_key, default)


app = cdk.App()

account = context_or_env(app, "account", "CDK_DEFAULT_ACCOUNT", "")
region = context_or_env(app, "region", "CDK_DEFAULT_REGION", "us-east-1")
rds_stack_name = context_or_env(
    app, "stackName", "CDK_STACK_NAME", "RealEstateRdsMvpStack"
)
fargate_stack_name = context_or_env(
    app, "fargateStackName", "CDK_FARGATE_STACK_NAME", "RealEstateFargateApiStack"
)
db_name = context_or_env(app, "dbName", "DB_NAME", "real_estate")
db_username = context_or_env(app, "dbUsername", "DB_USERNAME", "admin")
db_access_cidr = context_or_env(app, "dbAccessCidr", "DB_ACCESS_CIDR", "0.0.0.0/0")
public_access = (
    context_or_env(app, "publicAccess", "DB_PUBLIC_ACCESS", "true").lower() == "true"
)
enable_fargate = (
    context_or_env(app, "enableFargate", "ENABLE_FARGATE", "false").lower() == "true"
)
s3_bucket = context_or_env(app, "s3Bucket", "S3_BUCKET", "")
s3_prefix = context_or_env(app, "s3Prefix", "S3_PREFIX", "data/processed")
cors_allowed_origins = context_or_env(
    app, "corsAllowedOrigins", "CORS_ALLOWED_ORIGINS", "*"
)
cpu = int(context_or_env(app, "fargateCpu", "FARGATE_CPU", "2048"))
memory = int(context_or_env(app, "fargateMemory", "FARGATE_MEMORY", "4096"))
desired_count = int(
    context_or_env(app, "fargateDesiredCount", "FARGATE_DESIRED_COUNT", "1")
)

if not account:
    raise ValueError(
        "AWS account not resolved. Set CDK_DEFAULT_ACCOUNT or pass -c account=<id>."
    )

env = cdk.Environment(account=account, region=region)
repo_root = Path(__file__).resolve().parents[2]

rds_stack = RdsMvpStack(
    app,
    rds_stack_name,
    env=env,
    db_name=db_name,
    db_username=db_username,
    db_access_cidr=db_access_cidr,
    public_access=public_access,
)

if enable_fargate:
    if not s3_bucket:
        raise ValueError(
            "Fargate is enabled but S3 bucket is missing. "
            "Pass -c s3Bucket=<name> or set S3_BUCKET / ENABLE_FARGATE=false."
        )
    FargateApiStack(
        app,
        fargate_stack_name,
        env=env,
        vpc=rds_stack.vpc,
        db=rds_stack.db,
        db_security_group=rds_stack.db_security_group,
        db_secret=rds_stack.db_secret,
        db_name=db_name,
        s3_bucket_name=s3_bucket,
        s3_prefix=s3_prefix,
        docker_context=repo_root,
        cpu=cpu,
        memory_limit_mib=memory,
        desired_count=desired_count,
        cors_allowed_origins=cors_allowed_origins,
    )

app.synth()
