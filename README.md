# Multimodal document processing with Anthropic Claude 3 on Amazon Bedrock
<!--BEGIN STABILITY BANNER-->

---

![Stability: Experimental](https://img.shields.io/badge/stability-Experimental-important.svg?style=for-the-badge)

> All classes are under active development and subject to non-backward compatible changes or removal in any
> future version. These are not subject to the [Semantic Versioning](https://semver.org/) model.
> This means that while you may use them, you may need to update your source code when upgrading to a newer version of this package.

---
<!--END STABILITY BANNER-->


# Requirements
* [Create an AWS account](https://portal.aws.amazon.com/gp/aws/developer/registration/index.html) if you do not already have one and log in. The IAM user that you use must have sufficient permissions to make necessary AWS service calls and manage AWS resources.
* [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) installed and configured
* [Git Installed](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
* [AWS CDK Toolkit](https://docs.aws.amazon.com/cdk/latest/guide/cli.html) installed and configured
* [Python 3.9+](https://www.python.org/downloads/) installed

# Deployment
1. Create a new directory, navigate to that directory in a terminal and clone the GitHub repository:
    ```
    git clone https://github.com/TODO
    ```
2. Change directory to the pattern directory:
    ```
    cd fsi-idp-with-bedrock
    ```
3. Create a virtual environment for Python
    ```
    python3 -m venv .venv
    ```
4. Activate the virtual environment
    ```
    source .venv/bin/activate
    ```
    For a Windows platform, activate the virtualenv like this:
    ```
    .venv\Scripts\activate.bat
    ```
5. Install the Python required dependencies:
    ```
    pip install -r requirements.txt
    ```
6. Deploy the CDK stack:
    ```
    cdk deploy BedrockIDPClaude3Workflow
    ```

## Procedure

S3 File Upload
1. Create an environment variable in the AWS CLI for the S3 uploads bucket
```
export S3_UPLOADS_FOLDER=$(aws cloudformation list-exports --query 'Exports[?Name==`BedrockIDPClaude3Workflow-DocumentUploadLocation`].Value' --output text)
```

2. Copy a sample file to the S3 bucket to start the Step Functions execution
```
aws s3 cp ../fsi-idp-with-bedrock/docs/insurance_invoice.png $S3_UPLOADS_FOLDER
```






