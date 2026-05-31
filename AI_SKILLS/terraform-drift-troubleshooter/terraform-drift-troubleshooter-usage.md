# Terraform Drift Troubleshooter - Usage Guide

**Author:** Kartheek Anand

## What this skill does

This skill helps analyze Terraform troubleshooting cases with a focus on drift detection, state mismatch, refresh-only changes, and unexpected plan output.

It is designed to produce three outputs every time:

- a concise triage summary
- a step-by-step fix plan
- a full incident report

---

## How to use it

Paste either:

- a short incident summary, or
- raw Terraform output such as `terraform plan`, `terraform plan -refresh-only`, `terraform state show`, or error logs

A strong input usually includes:

- Terraform version
- provider version
- backend type
- workspace
- relevant plan output
- refresh-only output
- recent changes
- whether the resource was changed manually outside Terraform

---

## Example 1: drift detected in plan

### Input

    Terraform plan shows changes to aws_instance.web after someone modified tags manually in the console.
    Backend is remote state. No code change was made in this commit.

### Expected output

#### concise triage summary
- Issue: likely remote drift on a managed resource
- Drift type: state/remote drift
- Severity: medium
- Confidence: high
- Most likely cause: infrastructure changed outside Terraform
- Immediate next step: run `terraform plan -refresh-only` and compare state with provider values

#### step-by-step fix plan
1. Back up the current state.
2. Run `terraform plan -refresh-only` to confirm drift.
3. Compare `terraform state show aws_instance.web` with the provider-visible resource.
4. Decide whether to reconcile state only or apply the intended configuration change.
5. Re-run `terraform plan` and confirm the diff is expected.

#### full incident report
- Summary: manual tag changes caused Terraform drift.
- Evidence: plan showed tag differences without a corresponding code change.
- Root cause: resource was changed outside Terraform.
- Remediation: reconcile tags through Terraform and refresh state.
- Prevention: restrict manual console edits and monitor drift more frequently.

---

## Example 2: refresh-only shows changes

### Input

    `terraform plan -refresh-only` shows that the security group rules changed, but the configuration file was not edited.

### Expected output

#### concise triage summary
- Issue: confirmed drift between Terraform state and real infrastructure
- Drift type: state drift
- Severity: high
- Confidence: high
- Most likely cause: manual or external change to the security group
- Immediate next step: compare `terraform state show` with actual remote values before any apply

#### step-by-step fix plan
1. Confirm the plan is refresh-only.
2. Capture a state backup.
3. Inspect the state and remote resource values.
4. Identify which rule changes are intended.
5. Reconcile using the safest path, then validate with a normal plan.

#### full incident report
- Summary: refresh-only revealed unmanaged changes in the security group.
- Evidence: plan output diverged from the last known state.
- Root cause: out-of-band change.
- Remediation: update Terraform configuration or restore the original rule set.
- Prevention: limit manual edits and use drift checks in CI.

---

## Example 3: resource exists but is missing from state

### Input

    The database exists in OCI, but Terraform says it will create a new one. The resource is not present in state.

### Expected output

#### concise triage summary
- Issue: import gap
- Drift type: state mismatch due to unmanaged resource
- Severity: high
- Confidence: high
- Most likely cause: resource was created outside Terraform or state was lost
- Immediate next step: import the resource before planning further changes

#### step-by-step fix plan
1. Verify the real resource exists.
2. Confirm the resource address expected by Terraform.
3. Import the resource into state.
4. Run a new plan.
5. Resolve remaining differences.

#### full incident report
- Summary: Terraform does not know about an existing database.
- Evidence: real resource exists, but state does not include it.
- Root cause: import gap or state loss.
- Remediation: import and normalize configuration.
- Prevention: always create new managed resources through Terraform.

---

## What to ask for when evidence is missing

If the user has not pasted enough detail, ask only for the smallest useful set of evidence:

- `terraform plan` output
- `terraform plan -refresh-only` output
- `terraform state list`
- `terraform state show <address>`
- recent change description
- provider and backend details

---

## Best practice for using the skill

Use the skill as a triage and analysis assistant, then verify the suggested commands before changing anything in production.
