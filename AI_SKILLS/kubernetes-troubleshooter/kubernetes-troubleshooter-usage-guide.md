# Kubernetes Troubleshooter Skill - Usage Guide

**Author:** Kartheek Anand

## What this is

This guide explains how to use the Kubernetes Troubleshooter skill to analyze incidents from either:

- a short incident summary
- pasted `kubectl` output, logs, events, and manifests
- both together

The skill is self-contained and does not need connectors.

---

## Best way to use it

Give the skill:

- namespace
- workload name
- pod name if available
- exact error message
- recent changes
- `kubectl describe` output
- pod logs
- events
- deployment, service, ingress, PVC, or node details if relevant

Then ask for one of these outputs:

- a concise triage summary
- a step-by-step fix plan
- a full incident report
- all three together

---

## Recommended prompt pattern

Use a prompt like this:

    Act as a senior Kubernetes SRE.

    Troubleshoot this issue:
    <incident summary>

    Evidence:
    <paste kubectl output, logs, events, and manifest snippets>

    Give me:
    1. concise triage summary
    2. step-by-step fix plan
    3. full incident report

---

## Example incident

### Input

    Namespace: dev
    Pod: webapp-7d8f9c5f7f-abcde
    Symptom: CrashLoopBackOff after a ConfigMap change

    kubectl describe pod output:
    - Last State: Terminated
    - Reason: Error
    - Exit Code: 1
    - Readiness probe failed

    kubectl logs output:
    - Error reading config: missing DATABASE_URL

    Recent change:
    - ConfigMap updated before the restart

---

## Sample output 1: concise triage summary

    The pod is most likely crashing because the application cannot find the DATABASE_URL configuration value after the ConfigMap update.
    The strongest evidence is the application log showing a missing config error and the pod status showing Exit Code 1.
    Confidence: high.
    Immediate next check: verify the ConfigMap keys and the environment variable mapping in the deployment.

---

## Sample output 2: step-by-step fix plan

    1. Inspect the ConfigMap in the dev namespace.
       Command: kubectl get configmap -n dev

    2. Check the deployment environment variables.
       Command: kubectl describe deploy webapp -n dev

    3. Confirm the pod receives DATABASE_URL from the ConfigMap or Secret.
       Command: kubectl exec -n dev <pod> -- printenv | grep DATABASE_URL

    4. Fix the missing or renamed key in the ConfigMap.

    5. Restart the deployment.
       Command: kubectl rollout restart deploy/webapp -n dev

    6. Validate the pod becomes Ready and logs no longer show the missing config error.

---

## Sample output 3: full incident report

    # Incident Report: CrashLoopBackOff in webapp

    ## Impact
    The web application was unavailable in the dev namespace.

    ## Symptoms
    - Pod repeatedly restarted
    - Readiness probe failed
    - Application log reported a missing DATABASE_URL value

    ## Scope
    - Namespace: dev
    - Workload: webapp deployment

    ## Probable root cause
    The ConfigMap update removed or renamed a key required by the application.

    ## Evidence reviewed
    - Pod describe output
    - Container logs
    - Recent ConfigMap change

    ## Remediation
    - Restore the missing configuration key
    - Restart the deployment
    - Verify pod readiness

    ## Prevention
    - Add config validation at startup
    - Use a deployment checklist for ConfigMap changes
    - Add a pre-deploy smoke test

---

## How to get a concise triage summary only

Ask:

    Give me a concise triage summary only.

This is useful during incidents when you want a fast first pass.

---

## How to get a fix plan only

Ask:

    Give me a step-by-step fix plan.

This is useful when you already know the issue and want a safe recovery sequence.

---

## How to get a full incident report

Ask:

    Give me a full incident report with impact, root cause, remediation, and prevention.

This is useful for post-incident documentation and follow-up work.

---

## Good usage tips

- Start with the symptom.
- Paste the raw evidence.
- Include recent changes.
- Ask for likely root causes first.
- Ask for validation commands.
- Ask for a safe fix sequence.
- Use the full incident report after the issue is resolved.

---

## What makes the skill useful

The skill is useful because it turns Kubernetes troubleshooting into a repeatable workflow:

1. Collect evidence
2. Classify the failure
3. Identify the most likely root cause
4. Suggest the next commands
5. Propose a fix
6. Validate recovery
7. Write the incident report

---

## Key takeaway

Use the skill when you want structured Kubernetes troubleshooting from either a brief incident description, raw `kubectl` output, or both.

It is especially useful for SREs, DevOps engineers, and platform engineers who want a consistent incident triage format.

