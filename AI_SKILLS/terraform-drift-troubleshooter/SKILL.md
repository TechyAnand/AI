---
name: terraform-drift-troubleshooter
description: terraform troubleshooting for plan/apply failures, refresh-only drift detection, state mismatch, unexpected resource changes, import gaps, provider normalization diffs, and post-change validation. use when a user pastes terraform output, state, logs, or an incident summary and needs a concise triage summary, a step-by-step fix plan, or a full incident report. self-contained; no connectors required.
---

# Terraform Drift Troubleshooter

## Overview

Use this skill to diagnose Terraform issues with a strong focus on drift detection. Accept either a short incident summary or pasted Terraform evidence such as plan output, refresh-only output, state snippets, backend details, and error logs.

## Operating rules

- Start by classifying the problem into one or more of these buckets: config drift, state drift, import gap, provider normalization, backend/state access issue, or apply-time failure.
- Separate facts from inference.
- Prefer safe validation steps before suggesting reconciliation.
- If the evidence is incomplete, ask only for the minimum missing Terraform output needed to continue.
- When drift appears, explain whether it affects code, state, remote infrastructure, or all three.
- Highlight blast radius when a plan indicates create, replace, or destroy actions.

## Drift detection workflow

1. Identify the Terraform context: version, provider, backend, workspace, and recent changes.
2. Inspect the plan or incident summary for signs of drift.
3. Compare desired configuration with Terraform state.
4. Use refresh-only signals to tell state drift from config drift.
5. Identify whether the resource is managed, imported, tainted, or missing from state.
6. Propose a safe remediation path.
7. Verify the result and note prevention steps.

See `references/drift-workflow.md` for detailed signals and interpretation rules.

## What to look for first

Use the user's evidence to decide whether the issue is likely:

- Config drift: the Terraform code changed and now plans a real infrastructure update.
- State drift: the state file no longer matches reality.
- Remote drift: someone changed infrastructure outside Terraform.
- Import gap: the resource exists but is not in state.
- Provider normalization: harmless attribute formatting differences create noisy diffs.
- Backend issue: state lock, stale state, or remote backend access problem.

## Output format

Unless the user requests a different format, always provide these three sections in order:

### 1. concise triage summary
Include: issue type, drift type, severity, confidence, and the most likely cause.

### 2. step-by-step fix plan
Include the safest checks first, then the remediation path, then validation commands.

### 3. full incident report
Include: summary, evidence, analysis, root cause, remediation, prevention, and follow-up items.

## Fix-plan guidance

When giving remediation steps:

- Prefer read-only checks first.
- Recommend backups of state before risky changes.
- Distinguish state reconciliation from infrastructure changes.
- Call out when `terraform apply` would change real infrastructure.
- Call out when `terraform apply -refresh-only` would update state without changing infrastructure.
- Mention `terraform import` when a resource exists outside state.
- Mention provider version or lock-file drift when diffs look version-related.
- Mention `terraform state rm` only when the user explicitly wants to detach a resource from state.

## Useful commands to request or interpret

- `terraform version`
- `terraform init`
- `terraform validate`
- `terraform plan`
- `terraform plan -refresh-only`
- `terraform apply -refresh-only`
- `terraform show`
- `terraform state list`
- `terraform state show <address>`
- `terraform state pull`
- `terraform workspace show`
- `terraform providers`

## Writing style

- Be practical and direct.
- Prefer short headings and bullets.
- When you infer something, label it as an inference.
- When the evidence is weak, say so clearly.
- Avoid overclaiming the root cause if the plan or state evidence is incomplete.

## Example response shape

### concise triage summary
- Issue: likely state drift in a managed resource
- Severity: medium
- Confidence: high
- Why: refresh-only shows changes, while config did not change
- Immediate next step: validate provider-visible attributes and state contents

### step-by-step fix plan
1. Back up current state.
2. Run `terraform plan -refresh-only` to confirm drift.
3. Compare `terraform state show` with provider-visible resource data.
4. Reconcile using import or refresh-only, depending on whether the resource exists in state.
5. Re-run plan and validate that only expected changes remain.

### full incident report
- Summary
- Timeline
- Evidence
- Analysis
- Root cause
- Remediation
- Validation
- Prevention
