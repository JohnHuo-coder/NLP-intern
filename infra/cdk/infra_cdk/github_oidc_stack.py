from aws_cdk import CfnOutput, Stack, aws_iam as iam
from constructs import Construct

_GITHUB_ISSUER = "token.actions.githubusercontent.com"


class GitHubActionsOidcStack(Stack):
    """OIDC provider + role so GitHub Actions can assume CDK bootstrap roles."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        github_owner: str,
        github_repo: str,
        existing_provider: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        if not github_owner or not github_repo:
            raise ValueError("github_owner and github_repo are required.")

        if existing_provider:
            provider = iam.OpenIdConnectProvider.from_open_id_connect_provider_arn(
                self,
                "GitHubProvider",
                f"arn:aws:iam::{self.account}:oidc-provider/{_GITHUB_ISSUER}",
            )
        else:
            provider = iam.OpenIdConnectProvider(
                self,
                "GitHubProvider",
                url=f"https://{_GITHUB_ISSUER}",
                client_ids=["sts.amazonaws.com"],
            )

        role = iam.Role(
            self,
            "GitHubActionsRole",
            role_name="GitHubActionsNlpInternship",
            assumed_by=iam.FederatedPrincipal(
                provider.open_id_connect_provider_arn,
                {
                    "StringEquals": {
                        f"{_GITHUB_ISSUER}:aud": "sts.amazonaws.com",
                    },
                    "StringLike": {
                        f"{_GITHUB_ISSUER}:sub": (
                            f"repo:{github_owner}/{github_repo}:*"
                        ),
                    },
                },
                "sts:AssumeRoleWithWebIdentity",
            ),
            description="GitHub Actions OIDC role for NLP-intern CDK deploys",
        )
        role.add_to_policy(
            iam.PolicyStatement(
                sid="AssumeCdkBootstrapRoles",
                actions=["sts:AssumeRole", "sts:TagSession"],
                resources=[f"arn:aws:iam::{self.account}:role/cdk-hnb659fds-*"],
            )
        )

        CfnOutput(self, "RoleArn", value=role.role_arn)
        CfnOutput(self, "RoleName", value=role.role_name)
