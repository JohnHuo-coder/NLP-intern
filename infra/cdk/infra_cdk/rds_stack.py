from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_cloudwatch as cloudwatch,
    aws_ec2 as ec2,
    aws_rds as rds,
)
from constructs import Construct


class RdsMvpStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        db_name: str,
        db_username: str,
        db_access_cidr: str,
        public_access: bool,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.vpc = ec2.Vpc(
            self,
            "RdsVpc",
            max_azs=2,
            nat_gateways=0,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                ),
                ec2.SubnetConfiguration(
                    name="private-isolated",
                    subnet_type=ec2.SubnetType.PRIVATE_ISOLATED,
                    cidr_mask=24,
                ),
            ],
        )

        self.db_security_group = ec2.SecurityGroup(
            self,
            "RdsSecurityGroup",
            vpc=self.vpc,
            allow_all_outbound=True,
            description="Security group for RDS MySQL MVP instance",
        )
        self.db_security_group.add_ingress_rule(
            ec2.Peer.ipv4(db_access_cidr),
            ec2.Port.tcp(3306),
            f"MySQL access from {db_access_cidr}",
        )

        credentials = rds.Credentials.from_generated_secret(
            username=db_username,
            exclude_characters=' %+~`#$&*()|[]{}:;<>?!\'/@"\\',
        )

        subnet_type = (
            ec2.SubnetType.PUBLIC if public_access else ec2.SubnetType.PRIVATE_ISOLATED
        )

        self.db = rds.DatabaseInstance(
            self,
            "RealEstateMysql",
            engine=rds.DatabaseInstanceEngine.mysql(
                version=rds.MysqlEngineVersion.of("8.0", "8.0.46")
            ),
            instance_type=ec2.InstanceType.of(
                ec2.InstanceClass.BURSTABLE3,
                ec2.InstanceSize.MICRO,
            ),
            vpc=self.vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=subnet_type),
            publicly_accessible=public_access,
            credentials=credentials,
            database_name=db_name,
            allocated_storage=30,
            max_allocated_storage=100,
            storage_encrypted=True,
            multi_az=False,
            backup_retention=Duration.days(3),
            deletion_protection=False,
            removal_policy=RemovalPolicy.DESTROY,
            delete_automated_backups=True,
            security_groups=[self.db_security_group],
            auto_minor_version_upgrade=True,
        )
        self.db_secret = self.db.secret
        self.db_name = db_name

        # Basic RDS alarms (visible in CloudWatch; no SNS email wiring in MVP).
        cloudwatch.Alarm(
            self,
            "RdsCpuHighAlarm",
            metric=self.db.metric_cpu_utilization(
                period=Duration.minutes(5),
                statistic="Average",
            ),
            threshold=80,
            evaluation_periods=3,
            datapoints_to_alarm=2,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            alarm_description="RDS CPU >= 80% (2 of 3 five-minute periods)",
        )
        cloudwatch.Alarm(
            self,
            "RdsFreeStorageLowAlarm",
            metric=self.db.metric_free_storage_space(
                period=Duration.minutes(5),
                statistic="Average",
            ),
            # ~2 GiB remaining on a 30 GB volume
            threshold=2 * 1024 * 1024 * 1024,
            evaluation_periods=3,
            datapoints_to_alarm=2,
            comparison_operator=cloudwatch.ComparisonOperator.LESS_THAN_OR_EQUAL_TO_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            alarm_description="RDS free storage <= 2 GiB",
        )

        CfnOutput(self, "DbEndpoint", value=self.db.instance_endpoint.hostname)
        CfnOutput(self, "DbPort", value=str(self.db.instance_endpoint.port))
        CfnOutput(self, "DbName", value=db_name)
        CfnOutput(self, "DbSecretArn", value=self.db_secret.secret_arn)
        CfnOutput(
            self, "DbSecurityGroupId", value=self.db_security_group.security_group_id
        )
        CfnOutput(self, "VpcId", value=self.vpc.vpc_id)
