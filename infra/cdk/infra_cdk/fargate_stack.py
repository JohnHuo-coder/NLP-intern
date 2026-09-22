from pathlib import Path

from aws_cdk import (
    CfnOutput,
    Duration,
    Stack,
    aws_cloudwatch as cloudwatch,
    aws_cloudwatch_actions as cloudwatch_actions,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_elasticloadbalancingv2 as elbv2,
    aws_iam as iam,
    aws_logs as logs,
    aws_rds as rds,
    aws_s3 as s3,
    aws_secretsmanager as secretsmanager,
    aws_sns as sns,
)
from constructs import Construct


class FargateApiStack(Stack):
    """ALB + ECS Fargate service for the FastAPI search API (MVP)."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        vpc: ec2.IVpc,
        db: rds.IDatabaseInstance,
        db_security_group: ec2.ISecurityGroup,
        db_secret: secretsmanager.ISecret,
        db_name: str,
        s3_bucket_name: str,
        s3_prefix: str = "data/processed",
        docker_context: Path,
        cpu: int = 2048,
        memory_limit_mib: int = 4096,
        desired_count: int = 1,
        cors_allowed_origins: str = "*",
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        if not s3_bucket_name:
            raise ValueError("s3_bucket_name is required for FargateApiStack.")

        bucket = s3.Bucket.from_bucket_name(self, "ArtifactsBucket", s3_bucket_name)

        cluster = ecs.Cluster(self, "ApiCluster", vpc=vpc)

        task_sg = ec2.SecurityGroup(
            self,
            "ApiTaskSecurityGroup",
            vpc=vpc,
            allow_all_outbound=True,
            description="Security group for FastAPI Fargate tasks",
        )
        db_security_group.add_ingress_rule(
            peer=task_sg,
            connection=ec2.Port.tcp(3306),
            description="MySQL from Fargate API tasks",
        )

        # Public subnets + public IP: no NAT gateway needed for ECR/S3/Secrets.
        service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "ApiService",
            cluster=cluster,
            cpu=cpu,
            memory_limit_mib=memory_limit_mib,
            desired_count=desired_count,
            public_load_balancer=True,
            assign_public_ip=True,
            task_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            security_groups=[task_sg],
            circuit_breaker=ecs.DeploymentCircuitBreaker(rollback=True),
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_asset(
                    directory=str(docker_context),
                    file="Dockerfile",
                ),
                container_name="api",
                container_port=8000,
                enable_logging=True,
                log_driver=ecs.LogDrivers.aws_logs(
                    stream_prefix="api",
                    log_retention=logs.RetentionDays.ONE_WEEK,
                ),
                environment={
                    "AWS_REGION": Stack.of(self).region,
                    "AWS_DEFAULT_REGION": Stack.of(self).region,
                    "S3_BUCKET": s3_bucket_name,
                    "S3_PREFIX": s3_prefix,
                    "S3_PREFETCH_OPTIONAL": "true",
                    "MYSQLDATABASE": db_name,
                    "MYSQLHOST": db.instance_endpoint.hostname,
                    "MYSQLPORT": str(db.instance_endpoint.port),
                    "CORS_ALLOWED_ORIGINS": cors_allowed_origins,
                    "CORS_ALLOW_CREDENTIALS": "false",
                    "PORT": "8000",
                },
                secrets={
                    "MYSQLUSER": ecs.Secret.from_secrets_manager(
                        db_secret, field="username"
                    ),
                    "MYSQLPASSWORD": ecs.Secret.from_secrets_manager(
                        db_secret, field="password"
                    ),
                },
            ),
            health_check_grace_period=Duration.minutes(10),
        )

        service.target_group.configure_health_check(
            path="/health",
            healthy_http_codes="200",
            interval=Duration.seconds(30),
            timeout=Duration.seconds(5),
            healthy_threshold_count=2,
            unhealthy_threshold_count=5,
        )

        # Startup loads model + S3 artifacts; give the container more time.
        service.task_definition.default_container.add_ulimits(
            ecs.Ulimit(
                name=ecs.UlimitName.NOFILE,
                soft_limit=65536,
                hard_limit=65536,
            )
        )

        db_secret.grant_read(service.task_definition.execution_role)
        db_secret.grant_read(service.task_definition.task_role)
        bucket.grant_read(service.task_definition.task_role)

        # Explicit S3 list on prefix helps some SDK listing calls.
        service.task_definition.task_role.add_to_principal_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket"],
                resources=[bucket.bucket_arn],
                conditions={
                    "StringLike": {
                        "s3:prefix": [
                            f"{s3_prefix.strip('/')}/*",
                            s3_prefix.strip("/"),
                        ]
                    }
                },
            )
        )

        # --- CloudWatch: alarms + dashboard (MVP monitoring) ---
        alarm_topic = sns.Topic(
            self,
            "ApiAlarmTopic",
            display_name="RealEstate API alarms",
        )
        sns_action = cloudwatch_actions.SnsAction(alarm_topic)

        elb_5xx = service.load_balancer.metric_http_code_elb(
            code=elbv2.HttpCodeElb.ELB_5XX_COUNT,
            period=Duration.minutes(1),
            statistic="Sum",
        )
        target_5xx = service.target_group.metric_http_code_target(
            code=elbv2.HttpCodeTarget.TARGET_5XX_COUNT,
            period=Duration.minutes(1),
            statistic="Sum",
        )
        unhealthy_hosts = service.target_group.metric_unhealthy_host_count(
            period=Duration.minutes(1),
            statistic="Maximum",
        )
        cpu_util = service.service.metric_cpu_utilization(
            period=Duration.minutes(1),
            statistic="Average",
        )
        mem_util = service.service.metric_memory_utilization(
            period=Duration.minutes(1),
            statistic="Average",
        )
        p95_latency = service.target_group.metric_target_response_time(
            period=Duration.minutes(1),
            statistic="p95",
        )

        def _alarm(
            construct_id: str,
            *,
            metric: cloudwatch.IMetric,
            threshold: float,
            evaluation_periods: int,
            datapoints_to_alarm: int,
            comparison: cloudwatch.ComparisonOperator,
            treat_missing: cloudwatch.TreatMissingData,
            description: str,
        ) -> cloudwatch.Alarm:
            alarm = cloudwatch.Alarm(
                self,
                construct_id,
                metric=metric,
                threshold=threshold,
                evaluation_periods=evaluation_periods,
                datapoints_to_alarm=datapoints_to_alarm,
                comparison_operator=comparison,
                treat_missing_data=treat_missing,
                alarm_description=description,
            )
            alarm.add_alarm_action(sns_action)
            alarm.add_ok_action(sns_action)
            return alarm

        _alarm(
            "AlbElb5xxAlarm",
            metric=elb_5xx,
            threshold=5,
            evaluation_periods=3,
            datapoints_to_alarm=2,
            comparison=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            treat_missing=cloudwatch.TreatMissingData.NOT_BREACHING,
            description="ALB itself returning >=5 HTTP 5xx in a minute (2 of 3 periods)",
        )
        _alarm(
            "AlbTarget5xxAlarm",
            metric=target_5xx,
            threshold=5,
            evaluation_periods=3,
            datapoints_to_alarm=2,
            comparison=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            treat_missing=cloudwatch.TreatMissingData.NOT_BREACHING,
            description="API targets returning >=5 HTTP 5xx in a minute (2 of 3 periods)",
        )
        _alarm(
            "UnhealthyHostAlarm",
            metric=unhealthy_hosts,
            threshold=1,
            evaluation_periods=3,
            datapoints_to_alarm=3,
            comparison=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            treat_missing=cloudwatch.TreatMissingData.NOT_BREACHING,
            description="At least one unhealthy target for 3 consecutive minutes",
        )
        _alarm(
            "EcsCpuHighAlarm",
            metric=cpu_util,
            threshold=85,
            evaluation_periods=5,
            datapoints_to_alarm=3,
            comparison=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            treat_missing=cloudwatch.TreatMissingData.NOT_BREACHING,
            description="ECS CPU utilization >= 85% (3 of 5 minutes)",
        )
        _alarm(
            "EcsMemoryHighAlarm",
            metric=mem_util,
            threshold=85,
            evaluation_periods=5,
            datapoints_to_alarm=3,
            comparison=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            treat_missing=cloudwatch.TreatMissingData.NOT_BREACHING,
            description="ECS memory utilization >= 85% (3 of 5 minutes)",
        )

        dashboard = cloudwatch.Dashboard(
            self,
            "ApiDashboard",
            dashboard_name=f"{construct_id}-api",
        )
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="ALB / target 5xx",
                left=[elb_5xx, target_5xx],
                width=12,
            ),
            cloudwatch.GraphWidget(
                title="Unhealthy hosts",
                left=[unhealthy_hosts],
                width=12,
            ),
            cloudwatch.GraphWidget(
                title="ECS CPU %",
                left=[cpu_util],
                width=8,
                left_y_axis=cloudwatch.YAxisProps(min=0, max=100),
            ),
            cloudwatch.GraphWidget(
                title="ECS memory %",
                left=[mem_util],
                width=8,
                left_y_axis=cloudwatch.YAxisProps(min=0, max=100),
            ),
            cloudwatch.GraphWidget(
                title="Target p95 latency (s)",
                left=[p95_latency],
                width=8,
            ),
        )

        self.service = service
        self.task_security_group = task_sg
        self.alarm_topic = alarm_topic
        self.dashboard = dashboard

        CfnOutput(self, "ApiUrl", value=f"http://{service.load_balancer.load_balancer_dns_name}")
        CfnOutput(self, "LoadBalancerDns", value=service.load_balancer.load_balancer_dns_name)
        CfnOutput(self, "ClusterName", value=cluster.cluster_name)
        CfnOutput(self, "ServiceName", value=service.service.service_name)
        CfnOutput(self, "AlarmTopicArn", value=alarm_topic.topic_arn)
        CfnOutput(self, "DashboardName", value=dashboard.dashboard_name)
