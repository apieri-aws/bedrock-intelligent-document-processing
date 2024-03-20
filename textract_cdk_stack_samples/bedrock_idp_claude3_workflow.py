from constructs import Construct
import os
import re
import aws_cdk.aws_s3 as s3
import aws_cdk.aws_stepfunctions as sfn
import aws_cdk.aws_stepfunctions_tasks as tasks
import aws_cdk.aws_lambda as lambda_
import aws_cdk.aws_lambda_event_sources as eventsources
import aws_cdk.aws_iam as iam
import aws_cdk.aws_dynamodb as ddb
import amazon_textract_idp_cdk_constructs as tcdk
import cdk_nag as nag
from aws_cdk import CfnOutput, RemovalPolicy, Stack, Duration, Aws, Fn, Aspects
from aws_solutions_constructs.aws_lambda_opensearch import LambdaToOpenSearch
from aws_cdk import aws_opensearchservice as opensearch


class BedrockIDPClaude3Workflow(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(
            scope,
            construct_id,
            description="Information extraction using GenAI with Bedrock Claude3",
            **kwargs,
        )

        script_location = os.path.dirname(__file__)
        s3_upload_prefix = "uploads"
        s3_output_prefix = "textract-output"
        s3_csv_output_prefix = "csv-output"
        # s3_opensearch_output_prefix = "textract-opensearch-output"
        # s3_temp_output_prefix = "textract-temp"
        s3_split_document_prefix = "textract-split-documents"
        s3_txt_output_prefix = "textract-text-output"
        # s3_comprehend_output_prefix = "comprehend-output"
        s3_bedrock_classification_output_prefix = 'bedrock-classification-output'
        s3_bedrock_extraction_output_prefix = 'bedrock-extraction-output'

        workflow_name = "BedrockIDPClaude3"
        current_region = Stack.of(self).region
        account_id = Stack.of(self).account
        stack_name = Stack.of(self).stack_name

        #######################################
        # BEWARE! This is a demo/POC setup
        # Remove the auto_delete_objects=True and removal_policy=RemovalPolicy.DESTROY
        # when the documents should remain after deleting the CloudFormation stack!
        #######################################

        # Create the bucket for the documents and outputs
        document_bucket = s3.Bucket(
            self,
            f"{workflow_name}Bucket",
            removal_policy=RemovalPolicy.DESTROY,
            enforce_ssl=True,
            auto_delete_objects=False,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        )
        s3_output_bucket = document_bucket.bucket_name
        # get the event source that will be used later to trigger the executions
        s3_event_source = eventsources.S3EventSource(
            document_bucket,
            events=[s3.EventType.OBJECT_CREATED],
            filters=[s3.NotificationKeyFilter(prefix=s3_upload_prefix)],
        )

        # the decider checks if the document is of valid format and gets the
        # number of pages
        decider_task = tcdk.TextractPOCDecider(
            self,
            f"{workflow_name}-Decider",
            textract_decider_max_retries=10000,
            s3_input_bucket=document_bucket.bucket_name,
            s3_input_prefix=s3_upload_prefix,
        )
        # The splitter takes a document and splits into the max_number_of_pages_per_document
        # This is particulary useful when working with documents that exceed the Textract limits
        # or when the workflow requires per page processing
        document_splitter_task = tcdk.DocumentSplitter(
            self,
            "DocumentSplitter",
            s3_output_bucket=s3_output_bucket,
            s3_output_prefix=s3_split_document_prefix,
            s3_input_bucket=document_bucket.bucket_name,
            s3_input_prefix=s3_upload_prefix,
            max_number_of_pages_per_doc=1,
            lambda_log_level="INFO",
            textract_document_splitter_max_retries=10000,
        )

        # Call Textract sync
        textract_sync_task = tcdk.TextractGenericSyncSfnTask(
            self,
            "TextractSync",
            s3_output_bucket=document_bucket.bucket_name,
            s3_output_prefix=s3_output_prefix,
            integration_pattern=sfn.IntegrationPattern.WAIT_FOR_TASK_TOKEN,
            lambda_log_level="DEBUG",
            timeout=Duration.hours(24),
            input=sfn.TaskInput.from_object({
                "Token":
                sfn.JsonPath.task_token,
                "ExecutionId":
                sfn.JsonPath.string_at('$$.Execution.Id'),
                "Payload":
                sfn.JsonPath.entire_payload,
            }),
            result_path="$.textract_result")
            
        # Generates CSV data based on Texctract features defined in manifest file in ../lambda/start_with_all_features
        generate_csv = tcdk.TextractGenerateCSV(
            self,
            "GenerateCsvTask",
            csv_s3_output_bucket=document_bucket.bucket_name,
            csv_s3_output_prefix=s3_csv_output_prefix,
            s3_input_bucket=document_bucket.bucket_name,
            s3_input_prefix=s3_output_prefix,
            lambda_log_level="DEBUG",
            output_type='CSV',
            integration_pattern=sfn.IntegrationPattern.WAIT_FOR_TASK_TOKEN,
            input=sfn.TaskInput.from_object({
                "Token":
                sfn.JsonPath.task_token,
                "ExecutionId":
                sfn.JsonPath.string_at('$$.Execution.Id'),
                "Payload":
                sfn.JsonPath.entire_payload,
            }),
            result_path="$.csv_output_location")

        # Text generation task for classification
        generate_text = tcdk.TextractGenerateCSV(
            self,
            "GenerateText",
            csv_s3_output_bucket=document_bucket.bucket_name,
            csv_s3_output_prefix=s3_txt_output_prefix,
            output_type="LINES",
            lambda_log_level="DEBUG",
            integration_pattern=sfn.IntegrationPattern.WAIT_FOR_TASK_TOKEN,
            input=sfn.TaskInput.from_object(
                {
                    "Token": sfn.JsonPath.task_token,
                    "ExecutionId": sfn.JsonPath.string_at("$$.Execution.Id"),
                    "Payload": sfn.JsonPath.entire_payload,
                }
            ),
            result_path="$.txt_output_location",
        )

        # DynamoDB table for Bedrock configuration
        bedrock_table = ddb.Table(
            self,
            "BedrockPromptConfig",
            partition_key=ddb.Attribute(name="id", type=ddb.AttributeType.STRING),
            removal_policy=RemovalPolicy.DESTROY,  # Only for dev/test environments
            billing_mode=ddb.BillingMode.PAY_PER_REQUEST,  # Use on-demand billing mode
        )

        # Bedrock classification
        bedrock_idp_classification_function: lambda_.IFunction = lambda_.DockerImageFunction(  # type: ignore
            self,
            "BedrockIDPClassificationFunction",
            code=lambda_.DockerImageCode.from_image_asset(
                os.path.join(script_location, "../lambda/bedrock")
            ),
            memory_size=128,
            timeout=Duration.seconds(900),
            architecture=lambda_.Architecture.X86_64,
            environment={
                "LOG_LEVEL": "DEBUG",
                "FIXED_KEY": "CLASSIFICATION",
                "BEDROCK_CONFIGURATION_TABLE": bedrock_table.table_name,
                "S3_OUTPUT_PREFIX": s3_bedrock_classification_output_prefix,
                "S3_OUTPUT_BUCKET": document_bucket.bucket_name
            },
        )

        # Grant classification function permissions to DynamoDB table and Bedrock
        bedrock_table.grant_read_data(bedrock_idp_classification_function)
        document_bucket.grant_read_write(bedrock_idp_classification_function)
        bedrock_idp_classification_function.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],
                resources=["*"],
            )
        )

        # Create Bedrock classification task
        bedrock_idp_classification_task = tasks.LambdaInvoke(
            self,
            "BedrockIDPClassificationTask",
            lambda_function=bedrock_idp_classification_function,
            output_path="$.Payload",
        )

        bedrock_idp_classification_task.add_retry(
            max_attempts=10,
            errors=[
                "Lambda.TooManyRequestsException",
                "ModelNotReadyException",
                "ModelTimeoutException",
                "ServiceQuotaExceededException",
                "ThrottlingException",
            ],
        )

        # Create Bedrock extraction function
        bedrock_idp_extraction_function: lambda_.IFunction = lambda_.DockerImageFunction(  # type: ignore
            self,
            "BedrockIDPExtractionFunction",
            code=lambda_.DockerImageCode.from_image_asset(
                os.path.join(script_location, "../lambda/bedrock")
            ),
            memory_size=512,
            timeout=Duration.seconds(900),
            architecture=lambda_.Architecture.X86_64,
            environment={
                "LOG_LEVEL": "DEBUG",
                "BEDROCK_CONFIGURATION_TABLE": bedrock_table.table_name,
                "S3_OUTPUT_PREFIX": s3_bedrock_extraction_output_prefix,
                "S3_OUTPUT_BUCKET": document_bucket.bucket_name
            },
        )

        # Grant extraction function permissions to DynamoDB table and Bedrock
        bedrock_table.grant_read_data(bedrock_idp_extraction_function)
        document_bucket.grant_read_write(bedrock_idp_extraction_function)
        bedrock_idp_extraction_function.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],
                resources=["*"],
            )
        )

        # Create extraction task
        bedrock_idp_extraction_task = tasks.LambdaInvoke(
            self,
            "BedrockIDPExtractionTask",
            lambda_function=bedrock_idp_extraction_function,
            output_path="$.Payload",
        )

        bedrock_idp_extraction_task.add_retry(
            max_attempts=10,
            errors=[
                "Lambda.TooManyRequestsException",
                "ModelNotReadyException",
                "ModelTimeoutException",
                "ServiceQuotaExceededException",
                "ThrottlingException",
            ],
        )
        
        # Create function for Bedrock validity check
        bedrock_idp_validity_function: lambda_.IFunction = lambda_.DockerImageFunction(  # type: ignore
            self,
            "BedrockIDPValidityFunction",
            code=lambda_.DockerImageCode.from_image_asset(
                os.path.join(script_location, "../lambda/bedrock_document_validity")
            ),
            memory_size=512,
            timeout=Duration.seconds(900),
            architecture=lambda_.Architecture.X86_64,
            environment={
                "LOG_LEVEL": "DEBUG",
                "BEDROCK_CONFIGURATION_TABLE": bedrock_table.table_name,
                "S3_OUTPUT_PREFIX": s3_bedrock_extraction_output_prefix,
                "S3_OUTPUT_BUCKET": document_bucket.bucket_name
            },
        )

        # Grant extraction function permissions to DynamoDB table and Bedrock
        bedrock_table.grant_read_data(bedrock_idp_validity_function)
        document_bucket.grant_read_write(bedrock_idp_validity_function)
        bedrock_idp_validity_function.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],
                resources=["*"],
            )
        )

        # Create validity check task
        bedrock_idp_validity_check = tasks.LambdaInvoke(
            self,
            "BedrockIDPValidityCheck",
            lambda_function=bedrock_idp_validity_function,
            output_path="$.Payload",
        )

        bedrock_idp_validity_check.add_retry(
            max_attempts=10,
            errors=[
                "Lambda.TooManyRequestsException",
                "ModelNotReadyException",
                "ModelTimeoutException",
                "ServiceQuotaExceededException",
                "ThrottlingException",
            ],
        )

        # Function to add results from Bedrock classification into the step function context
        bedrock_idp_classification_context_function: lambda_.IFunction = lambda_.DockerImageFunction(  # type: ignore
            self,
            "BedrockIDPClassificationContextFunction",
            code=lambda_.DockerImageCode.from_image_asset(
                os.path.join(script_location, "../lambda/bedrock_print_content")
            ),
            memory_size=128,
            timeout=Duration.seconds(900),
            architecture=lambda_.Architecture.X86_64,
            environment={
                "LOG_LEVEL": "DEBUG",
            },
        )

        bedrock_idp_classification_context_task = tasks.LambdaInvoke(
            self,
            "BedrockIDPClassificationContextTask",
            lambda_function=bedrock_idp_classification_context_function,
            output_path="$.Payload",
        )

        # Grant classification context function permissions to DynamoDB table
        document_bucket.grant_read_write(bedrock_idp_classification_context_function)

        # Creating the StepFunction workflow
        # Determine if the input file is a supported image type for Claude 3
        is_supported_image_type = sfn.Condition.or_(
            sfn.Condition.string_equals("$.mime", "image/jpeg"),
            sfn.Condition.string_equals("$.mime", "image/png"),
        )
        
        is_supported_image_choice = (
            sfn.Choice(self, "IsSupportedImage")
            .when(is_supported_image_type, bedrock_idp_validity_check)
            .otherwise(sfn.Pass(self, "Not supported image type"))
        )
        
        # Route StepFunction based on Bedrock classification
        bedrock_idp_extraction_parallel = (
            sfn.Parallel(self, "parallel_extraction")
            .branch(bedrock_idp_extraction_task)
            .branch(is_supported_image_choice)
        )
        
        doc_type_choice = (
            sfn.Choice(self, "RouteDocType")
            .when(
                sfn.Condition.string_equals("$.classification.documentType", "BANK_STATEMENT"),
                bedrock_idp_extraction_parallel
            )
            .when(
                sfn.Condition.string_equals("$.classification.documentType", "BIRTH_CERTIFICATE"),
                bedrock_idp_extraction_parallel
            )            
            .when(
                sfn.Condition.string_equals("$.classification.documentType", "PAYSTUB"),
                bedrock_idp_extraction_parallel
            )
            .otherwise(sfn.Pass(self, "No processing"))
        )
        
        # Create chains for parallel_tasks
        form_table_chain = (
            sfn.Chain.start(generate_csv)
        )

        bedrock_chain = (
            sfn.Chain.start(generate_text)
            .next(bedrock_idp_classification_task)
            # .next(bedrock_idp_classification_context_task)
            .next(doc_type_choice)
        )

        parallel_tasks = (
            sfn.Parallel(self, "parallel")
            .branch(form_table_chain)
            .branch(bedrock_chain)
        )

        textract_sync_task.next(parallel_tasks)

        # Define map state to iterate on each page in the document
        map = sfn.Map(
            self,
            "Map State",
            items_path=sfn.JsonPath.string_at("$.pages"),
            parameters={
                "manifest": {
                    "s3Path": sfn.JsonPath.string_at(
                        "States.Format('s3://{}/{}/{}', \
                        $.documentSplitterS3OutputBucket, \
                        $.documentSplitterS3OutputPath, \
                        $$.Map.Item.Value)"
                    )
                },
                "mime": sfn.JsonPath.string_at("$.mime"),
                "originFileURI": sfn.JsonPath.string_at("$.originFileURI"),
            },
        )

        map.iterator(textract_sync_task)

        # Define workflow chain before map state
        workflow_chain = (
            sfn.Chain.start(decider_task).next(document_splitter_task).next(map)
        )

        # GENERIC
        state_machine = sfn.StateMachine(self, workflow_name, definition=workflow_chain)

        # The StartThrottle triggers based on event_source (in this case S3 OBJECT_CREATED)
        # and handles all the complexity of making sure the limits or bottlenecks are not exceeded
        sf_executions_start_throttle = tcdk.SFExecutionsStartThrottle(
            self,
            "ExecutionThrottle",
            state_machine_arn=state_machine.state_machine_arn,
            s3_input_bucket=document_bucket.bucket_name,
            s3_input_prefix=s3_upload_prefix,
            executions_concurrency_threshold=550,
            sqs_batch=10,
            lambda_log_level="INFO",
            event_source=[s3_event_source],
        )
        queue_url_urlencoded = ""
        if sf_executions_start_throttle.document_queue:
            # urlencode the SQS Queue link, otherwise the deep linking does not work properly.
            queue_url_urlencoded = Fn.join(
                "%2F",
                Fn.split(
                    "/",
                    Fn.join(
                        "%3A",
                        Fn.split(
                            ":", sf_executions_start_throttle.document_queue.queue_url
                        ),
                    ),
                ),
            )
        
        # OUTPUT
        CfnOutput(
            self,
            "DocumentUploadLocation",
            value=f"s3://{document_bucket.bucket_name}/{s3_upload_prefix}/",
            export_name=f"{Aws.STACK_NAME}-DocumentUploadLocation",
        )
        CfnOutput(
            self,
            "BedrockConfigurationTableName",
            value=bedrock_table.table_name,
            export_name=f"{Aws.STACK_NAME}-BedrockConfigurationTableName",
        )
        CfnOutput(
            self,
            "StepFunctionFlowLink",
            value=f"https://{current_region}.console.aws.amazon.com/states/home?region={current_region}#/statemachines/view/{state_machine.state_machine_arn}",  # noqa: E501
        ),
        CfnOutput(
            self,
            "DocumentQueueLink",
            value=f"https://{current_region}.console.aws.amazon.com/sqs/v2/home?region={current_region}#/queues/{queue_url_urlencoded}",  # noqa: E501
        )
        CfnOutput(
            self,
            "BedrockConfigurationTable",
            value=f"https://{current_region}.console.aws.amazon.com/dynamodbv2/home?region={current_region}#tables:selected={bedrock_table.table_name}",  # noqa: E501
        )

        # # NAG suppressions
        # nag.NagSuppressions.add_resource_suppressions(
        #     document_bucket,
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-S1",
        #                 reason="no server access log for this demo",
        #             )
        #         )
        #     ],
        # )
        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/{workflow_name}-Decider/TextractDecider/ServiceRole/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-IAM4",
        #                 reason="using AWSLambdaBasicExecutionRole",
        #             )
        #         )
        #     ],
        # )
        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/{workflow_name}-Decider/TextractDecider/ServiceRole/DefaultPolicy/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-IAM5",
        #                 reason="wildcard permission is for everything under prefix",
        #             )
        #         )
        #     ],
        # )

        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/DocumentSplitter/DocumentSplitterFunction/ServiceRole/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-IAM4",
        #                 reason="using AWSLambdaBasicExecutionRole",
        #             )
        #         )
        #     ],
        # )
        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/DocumentSplitter/DocumentSplitterFunction/ServiceRole/DefaultPolicy/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-IAM5",
        #                 reason="wildcard permission is for everything under prefix",
        #             )
        #         )
        #     ],
        # )

        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/TextractAsync/TextractTaskTokenTable/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-DDB3",
        #                 reason="no point-in-time-recovery required for demo",
        #             )
        #         )
        #     ],
        # )
        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/TextractAsync/TextractAsyncSNSRole/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-IAM4",
        #                 reason="following Textract SNS best practices",
        #             )
        #         )
        #     ],
        # )
        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/TextractAsync/TextractAsyncSNS/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-SNS3", reason="publisher is only Textract"
        #             )
        #         ),
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-SNS2", reason="no SNS encryption for demo"
        #             )
        #         ),
        #     ],
        # )
        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/TextractAsync/TextractAsyncCall/ServiceRole/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-IAM4",
        #                 reason="using AWSLambdaBasicExecutionRole",
        #             )
        #         )
        #     ],
        # )
        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/TextractAsync/TextractAsyncCall/ServiceRole/DefaultPolicy/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-IAM5",
        #                 reason="access only for bucket and prefix",
        #             )
        #         )
        #     ],
        # )
        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/TextractAsync/TextractAsyncSNSFunction/ServiceRole/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-IAM4",
        #                 reason="using AWSLambdaBasicExecutionRole",
        #             )
        #         )
        #     ],
        # )
        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/TextractAsync/TextractAsyncSNSFunction/ServiceRole/DefaultPolicy/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-IAM5",
        #                 reason="access only for bucket and prefix and state machine \
        #                     does not allow for resource constrain",
        #             )
        #         )
        #     ],
        # )
        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/TextractAsync/StateMachine/Role/DefaultPolicy/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-IAM5",
        #                 reason="access only for bucket and prefix and state machine \
        #                     does not allow for resource constrain",
        #             )
        #         )
        #     ],
        # )
        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/TextractAsync/StateMachine/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-SF1",
        #                 reason="no logging for StateMachine for demo",
        #             )
        #         ),
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-SF2", reason="no X-Ray logging for demo"
        #             )
        #         ),
        #     ],
        # )

        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/TextractAsyncToJSON2/TextractAsyncToJSON/ServiceRole/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-IAM4",
        #                 reason="using AWSLambdaBasicExecutionRole",
        #             )
        #         )
        #     ],
        # )
        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/TextractAsyncToJSON2/TextractAsyncToJSON/ServiceRole/DefaultPolicy/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-IAM5",
        #                 reason="wildcard permission is for everything under prefix",
        #             )
        #         )
        #     ],
        # )

        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/GenerateOpenSearchBatch/TextractCSVGenerator/ServiceRole/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-IAM4",
        #                 reason="using AWSLambdaBasicExecutionRole",
        #             )
        #         )
        #     ],
        # )
        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/GenerateOpenSearchBatch/TextractCSVGenerator/ServiceRole/DefaultPolicy/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-IAM5",
        #                 reason="wildcard permission is for everything under prefix",
        #             )
        #         )
        #     ],
        # )
        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/GenerateOpenSearchBatch/StateMachine/Role/DefaultPolicy/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-IAM5",
        #                 reason="wildcard permission is for everything under prefix",
        #             )
        #         )
        #     ],
        # )
        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/GenerateOpenSearchBatch/StateMachine/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-SF1",
        #                 reason="no logging for StateMachine for demo",
        #             )
        #         ),
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-SF2", reason="no X-Ray logging for demo"
        #             )
        #         ),
        #     ],
        # )

        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/LambdaOpenSearchPush/ServiceRole/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-IAM4",
        #                 reason="using AWSLambdaBasicExecutionRole",
        #             )
        #         )
        #     ],
        # )
        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/LambdaOpenSearchPush/ServiceRole/DefaultPolicy/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-IAM5",
        #                 reason="wildcard permission is for everything under prefix",
        #             )
        #         )
        #     ],
        # )

        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/OpenSearchResources/CognitoUserPool/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-COG1", reason="no password policy for demo"
        #             )
        #         ),
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-COG2", reason="no MFA for demo"
        #             )
        #         ),
        #     ],
        # )

        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/OpenSearchResources/CognitoAuthorizedRole/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-IAM5", reason="wildcard for es:ESHttp*"
        #             )
        #         )
        #     ],
        # )

        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/OpenSearchResources/OpenSearchDomain",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-OS1", reason="no VPC for demo"
        #             )
        #         ),
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-OS3",
        #                 reason="users have to be authorized to access, not limit on IP for demo",
        #             )
        #         ),
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-OS4", reason="no dedicated master for demo"
        #             )
        #         ),
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-OS7", reason="no zone awareness for demo"
        #             )
        #         ),
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-OS9",
        #                 reason="no minimally publish SEARCH_SLOW_LOGS and INDEX_SLOW_LOGS to CloudWatch Logs.",
        #             )
        #         ),
        #     ],
        # )
        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/LambdaOpenSearchMapping/ServiceRole/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-IAM4",
        #                 reason="using AWSLambdaBasicExecutionRole",
        #             )
        #         )
        #     ],
        # )

        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/SetMetaDataFunction/ServiceRole/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-IAM4",
        #                 reason="using AWSLambdaBasicExecutionRole",
        #             )
        #         )
        #     ],
        # )

        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/{workflow_name}/Role/DefaultPolicy/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-IAM5",
        #                 reason="limited to lambda:InvokeFunction for the Lambda functions used in the workflow",
        #             )
        #         )
        #     ],
        # )
        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/{workflow_name}/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-SF1",
        #                 reason="no logging for StateMachine for demo",
        #             )
        #         ),
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-SF2", reason="no X-Ray logging for demo"
        #             )
        #         ),
        #     ],
        # )

        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/ExecutionThrottle/DocumentQueue/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-SQS3",
        #                 reason="no DLQ required by design, DDB to show status of processing",
        #             )
        #         ),
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-SQS4", reason="no SSL forcing for demo"
        #             )
        #         ),
        #     ],
        # )
        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/ExecutionThrottle/IDPDocumentStatusTable/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-DDB3",
        #                 reason="no DDB point in time recovery for demo",
        #             )
        #         ),
        #     ],
        # )
        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/ExecutionThrottle/IDPExecutionsCounterTable/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-DDB3",
        #                 reason="no DDB point in time recovery for demo",
        #             )
        #         ),
        #     ],
        # )

        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/ExecutionThrottle/ExecutionsStartThrottle/ServiceRole/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-IAM4",
        #                 reason="using AWSLambdaBasicExecutionRole",
        #             )
        #         )
        #     ],
        # )
        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/ExecutionThrottle/ExecutionsStartThrottle/ServiceRole/DefaultPolicy/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-IAM5",
        #                 reason="wildcard permission is for everything under prefix",
        #             )
        #         )
        #     ],
        # )
        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/ExecutionThrottle/ExecutionsQueueWorker/ServiceRole/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-IAM4",
        #                 reason="using AWSLambdaBasicExecutionRole",
        #             )
        #         )
        #     ],
        # )
        # nag.NagSuppressions.add_resource_suppressions_by_path(
        #     stack=self,
        #     path=f"{stack_name}/ExecutionThrottle/ExecutionsThrottleCounterReset/ServiceRole/Resource",
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-IAM4",
        #                 reason="using AWSLambdaBasicExecutionRole",
        #             )
        #         )
        #     ],
        # )

        # nag.NagSuppressions.add_stack_suppressions(
        #     stack=self,
        #     suppressions=[
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-IAM4",
        #                 reason="using AWSLambdaBasicExecutionRole",
        #             )
        #         ),
        #         (
        #             nag.NagPackSuppression(
        #                 id="AwsSolutions-IAM5",
        #                 reason="internal CDK to set bucket notifications: https://github.com/aws/aws-cdk/issues/9552 ",
        #             )
        #         ),
        #     ],
        # )

        # Aspects.of(self).add(nag.AwsSolutionsChecks())
