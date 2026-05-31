# Drift Detection Workflow and Output Templates

## Drift taxonomy

### Config drift
The Terraform configuration changed and now plans infrastructure updates.

### State drift
Terraform state no longer matches the real infrastructure.

### Remote drift
Someone changed the infrastructure outside Terraform.

### Import gap
The resource exists in the target environment but is not tracked in state.

### Provider normalization
A harmless formatting or canonicalization difference causes noisy diffs.

### Backend issue
The state backend is stale, locked, unavailable, or inconsistent.

## Interpretation rules

- If `terraform plan` shows changes and the code changed, treat it as config-driven change first.
- If `terraform plan -refresh-only` shows changes but the code did not change, treat it as drift in state or remote infrastructure.
- If `terraform state list` does not include a real resource, suspect an import gap.
- If the diff is only cosmetic, suspect provider normalization.
- If Terraform cannot read or lock state, treat it as a backend issue before any reconciliation.

## Suggested evidence to request

- Terraform version
- Provider version
- Backend type
- Workspace
- Relevant plan output
- `terraform plan -refresh-only` output
- `terraform state list`
- `terraform state show <address>`
- Recent manual changes
- CI/CD change that may have touched infra

## Safe remediation ladder

1. Confirm the backend and lock status.
2. Back up state.
3. Verify whether the drift is real or only cosmetic.
4. Reconcile state with refresh-only if no infrastructure change is needed.
5. Import unmanaged resources if they already exist.
6. Apply only the minimum required configuration change.
7. Re-run plan and confirm the drift is gone.

## Concise triage summary template

- Issue: <one-line issue>
- Drift type: <config/state/remote/import/provider/backend>
- Severity: <low/medium/high>
- Confidence: <low/medium/high>
- Most likely cause: <cause>
- Immediate next step: <next step>

## Step-by-step fix plan template

1. Confirm the evidence.
2. Identify the drift type.
3. Validate the state and provider view.
4. Reconcile safely.
5. Re-run plan.
6. Confirm the final state.

## Full incident report template

- Executive summary
- Symptoms
- Evidence
- Drift analysis
- Root cause
- Fix applied or recommended
- Validation
- Prevention
- Follow-up actions
