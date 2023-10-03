data "aws_iam_policy_document" "hello" {
  statement {
    sid       = ""
    effect    = "Allow"
    resources = ["*"]
    actions   = ["redshift-serverless:*"]
  }

  statement {
    sid       = ""
    effect    = "Allow"
    resources = ["*"]
    actions   = ["sqlworkbench:*"]
  }