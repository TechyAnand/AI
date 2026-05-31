# Kubernetes Troubleshooting Guide with GenAI Prompts

**Author:** Kartheek Anand

## Purpose

This guide is designed for SREs, DevOps engineers, and platform engineers who want a structured way to troubleshoot Kubernetes issues using Generative AI prompts.

It is written as a practical operating guide that can be used directly during incidents, investigations, reviews, and learning sessions.

---

## Should this be a Markdown guide or an AI Skill?

For most teams, a **Markdown guide is the best first choice** because it is:

- easy to store in SCM
- easy to review and edit
- easy to share in runbooks or internal docs
- useful even without special tooling
- good for incident response and knowledge transfer

An **AI Skill** becomes useful later if you want:

- a reusable AI assistant with a fixed workflow
- consistent troubleshooting behavior
- a guided prompt flow inside ChatGPT
- automation around repeated troubleshooting patterns

### Recommendation

Start with this Markdown guide first.

Move it into an AI Skill later only if the team repeatedly uses the same troubleshooting process and wants a more interactive assistant.

---

# How to Use This Guide

Use this guide in the following order:

1. Collect symptoms
2. Ask the model for a first-pass diagnosis
3. Validate the likely cause with kubectl and logs
4. Narrow down the failure domain
5. Propose fixes
6. Verify recovery
7. Capture the root cause and follow-up actions

---

# Troubleshooting Mindset

Before asking GenAI for help, always gather:

- namespace
- workload name
- pod name
- exact error message
- recent changes
- deployment method
- time of failure
- cluster environment
- affected services
- logs and events

The better the context, the better the response.

---

# Core Troubleshooting Workflow

## Step 1: Confirm the symptom

Ask:

- What is failing?
- Since when?
- Is it all pods or only some?
- Is the issue intermittent or constant?
- Did anything change recently?

Useful commands:

    kubectl get pods -A
    kubectl get deploy -A
    kubectl get svc -A
    kubectl get events -A --sort-by=.metadata.creationTimestamp

---

## Step 2: Identify the failure layer

Classify the issue into one of these layers:

- Application
- Container image
- Pod scheduling
- Node health
- Service discovery
- Ingress
- DNS
- Storage
- Network policy
- RBAC
- Control plane

This helps reduce guesswork.

---

## Step 3: Inspect the workload

Useful commands:

    kubectl describe pod <pod-name> -n <namespace>
    kubectl logs <pod-name> -n <namespace>
    kubectl logs <pod-name> -n <namespace> --previous
    kubectl get pod <pod-name> -n <namespace> -o wide

---

## Step 4: Inspect events

Events often reveal the real cause.

Useful command:

    kubectl get events -n <namespace> --sort-by=.metadata.creationTimestamp

Look for:

- ImagePullBackOff
- CrashLoopBackOff
- FailedScheduling
- OOMKilled
- Readiness probe failures
- Volume mount failures
- Permission denied
- RBAC denied
- DNS resolution errors

---

## Step 5: Validate the fix

After any change, verify:

- pod status
- logs
- service endpoints
- application response
- ingress routing
- readiness and liveness probes

Useful commands:

    kubectl get pods -n <namespace>
    kubectl get endpoints -n <namespace>
    kubectl describe svc <service-name> -n <namespace>
    kubectl rollout status deploy/<deployment-name> -n <namespace>

---

# A Good Prompt Template for Troubleshooting

Use this pattern:

    Act as a senior Kubernetes SRE.

    Troubleshoot the following issue:
    <problem description>

    Environment:
    - Kubernetes version:
    - Namespace:
    - Workload type:
    - Cloud/on-prem:
    - Recent changes:

    Evidence:
    - kubectl describe output:
    - pod logs:
    - events:
    - service/ingress details:
    - node status:

    Output format:
    1. Likely root causes
    2. Recommended checks
    3. Probable fix
    4. Validation steps
    5. Prevention advice

---

# Symptom-Based GenAI Prompt Library

## 1. Pod in CrashLoopBackOff

Prompt:

    Act as a Kubernetes SRE.

    My pod is in CrashLoopBackOff.

    Here is the context:
    - Namespace: <namespace>
    - Pod name: <pod-name>
    - Deployment name: <deployment-name>

    Evidence:
    - kubectl describe pod output:
    - kubectl logs output:
    - recent deployment changes:
    - readiness/liveness probe settings:

    Please provide:
    1. probable root causes
    2. exact kubectl commands to verify each cause
    3. likely fix options
    4. what to check after redeploying

Common causes to consider:

- app exits immediately
- bad config
- missing secrets
- probe failures
- port mismatch
- command/entrypoint error

---

## 2. Pod stuck in Pending

Prompt:

    Act as a platform engineer.

    My pod is stuck in Pending.

    Context:
    - Namespace: <namespace>
    - Pod name: <pod-name>
    - Node count:
    - Resource requests/limits:
    - Storage class:
    - Taints/tolerations:

    Evidence:
    - kubectl describe pod:
    - kubectl get nodes:
    - kubectl get events:
    - storage and scheduling details:

    Please analyze whether the likely cause is:
    - insufficient CPU/memory
    - node selector mismatch
    - taints and tolerations
    - PVC binding failure
    - affinity rules
    - cluster capacity

---

## 3. ImagePullBackOff

Prompt:

    Act as a DevOps engineer.

    A Kubernetes pod is failing with ImagePullBackOff.

    Context:
    - Image name:
    - Registry:
    - Namespace:
    - Image pull secret:
    - Service account:

    Evidence:
    - kubectl describe pod:
    - image reference:
    - registry auth details:
    - recent image tag changes:

    Please provide:
    1. probable causes
    2. verification commands
    3. remediation steps
    4. prevention tips

Common causes:

- wrong image name
- wrong tag
- registry auth failure
- network access issue
- private registry secret missing

---

## 4. Service has no endpoints

Prompt:

    Act as a Kubernetes troubleshooting assistant.

    My service is reachable, but it has no endpoints.

    Context:
    - Namespace:
    - Service name:
    - Deployment name:
    - Selector labels:
    - Pod labels:

    Evidence:
    - kubectl describe svc:
    - kubectl get pods --show-labels:
    - kubectl get endpoints:
    - kubectl get endpointslice:

    Identify label mismatch, pod readiness issues, or selector problems.

---

## 5. Ingress returns 404 or 502

Prompt:

    Act as an SRE.

    My Kubernetes ingress is returning 404 or 502.

    Context:
    - Ingress name:
    - Namespace:
    - Ingress controller:
    - Service name:
    - Service port:
    - Application port:

    Evidence:
    - kubectl describe ingress:
    - kubectl get svc:
    - ingress controller logs:
    - application logs:
    - endpoint status:

    Please provide:
    1. likely root causes
    2. exact validation steps
    3. fix plan
    4. post-fix checks

---

## 6. Node NotReady

Prompt:

    Act as a Kubernetes platform engineer.

    One of my nodes is in NotReady state.

    Context:
    - Node name:
    - Cluster size:
    - CNI plugin:
    - kubelet status:
    - recent node changes:

    Evidence:
    - kubectl describe node:
    - kubectl get events:
    - node system logs:
    - kubelet logs:

    Please assess:
    - network plugin failure
    - disk pressure
    - memory pressure
    - kubelet failure
    - control plane connectivity
    - certificate issues

---

## 7. DNS Resolution Failure

Prompt:

    Act as a Kubernetes SRE.

    My pod cannot resolve DNS names.

    Context:
    - Namespace:
    - Pod name:
    - DNS query failing:
    - CoreDNS version:
    - network policy status:

    Evidence:
    - pod logs
    - /etc/resolv.conf from the pod
    - CoreDNS logs
    - kubectl get svc kube-dns or coredns
    - kubectl get endpoints in kube-system

    Please diagnose the DNS failure path.

---

## 8. Persistent Volume or PVC issues

Prompt:

    Act as a Kubernetes storage engineer.

    My pod cannot start because the PVC is not bound or the volume mount is failing.

    Context:
    - Namespace:
    - PVC name:
    - StorageClass:
    - PV details:
    - workload name:

    Evidence:
    - kubectl describe pvc:
    - kubectl describe pv:
    - kubectl describe pod:
    - events:
    - storage provisioner logs:

    Please identify whether this is:
    - storage class issue
    - provisioning issue
    - access mode mismatch
    - node attach issue
    - filesystem permission issue

---

# Incident Triage Prompt

Use this when you need a first-pass analysis quickly.

    Act as a senior Kubernetes SRE.

    I will share Kubernetes symptoms and evidence.

    Your job is to:
    - identify the most likely cause
    - classify the issue by layer
    - list the top 5 commands to confirm it
    - propose a safe remediation
    - mention any risks before applying the fix

    Keep the answer concise, practical, and operational.

---

# Deep-Dive Prompt

Use this when the issue is more complex.

    Act as a principal platform engineer.

    Analyze the Kubernetes issue in detail.

    Inputs:
    - symptoms
    - kubectl output
    - logs
    - events
    - recent changes
    - deployment manifests
    - node information
    - service and ingress details

    Deliver:
    1. executive summary
    2. probable root cause
    3. evidence supporting the hypothesis
    4. commands to validate
    5. remediation steps
    6. rollback options
    7. prevention and hardening recommendations

---

# Useful kubectl Commands

Inspect pods:

    kubectl get pods -n <namespace>
    kubectl describe pod <pod-name> -n <namespace>

View logs:

    kubectl logs <pod-name> -n <namespace>
    kubectl logs <pod-name> -n <namespace> --previous

Check deployments:

    kubectl get deploy -n <namespace>
    kubectl rollout status deploy/<deployment-name> -n <namespace>
    kubectl rollout history deploy/<deployment-name> -n <namespace>

Check service discovery:

    kubectl get svc -n <namespace>
    kubectl get endpoints -n <namespace>

Check nodes:

    kubectl get nodes
    kubectl describe node <node-name>

Check events:

    kubectl get events -n <namespace> --sort-by=.metadata.creationTimestamp

---

# How to Use GenAI Effectively

## Give the model the facts

Include:

- exact error message
- namespace
- pod name
- deployment name
- recent change
- logs
- events
- manifest snippet

## Ask for structured output

For example:

- root cause
- validation
- fix
- prevention

## Avoid vague prompts

Bad:

    Kubernetes is broken

Better:

    My pod in namespace dev is stuck in CrashLoopBackOff after a config map update.

## Ask for safe actions

Tell the model to avoid destructive commands unless necessary.

---

# Troubleshooting Checklist

Before asking GenAI, collect:

- `kubectl describe`
- `kubectl logs`
- events
- deployment YAML
- service YAML
- ingress YAML
- node status
- recent changes
- resource requests and limits
- network and DNS details

This makes the answer much better.

---

# Best Practices

1. Start with the symptom.
2. Add exact context.
3. Share logs and events.
4. Ask for likely causes first.
5. Ask for verification commands.
6. Ask for a safe fix.
7. Validate after applying the fix.
8. Capture the root cause.
9. Update the runbook.
10. Reuse the prompt template.

---

# Common Mistakes

## Too little context

Bad:

    My pod is failing. Help.

## No evidence

Bad:

    Something is wrong in Kubernetes.

## Asking for a fix without diagnosis

Bad:

    Give me a command to fix it.

## Not checking events

Events often reveal the real cause.

---

# Real-World Usage Pattern

A good operational flow is:

    Collect symptoms
        |
        v
    Paste evidence into GenAI
        |
        v
    Get likely causes
        |
        v
    Verify with kubectl
        |
        v
    Apply safe fix
        |
        v
    Validate recovery
        |
        v
    Document lessons learned

---

# Interview Questions

## What is Kubernetes troubleshooting with GenAI?

It is the use of AI prompts to analyze Kubernetes symptoms, logs, and events to help diagnose and fix issues faster.

## Why is structured prompting important?

Because better context produces more accurate and useful troubleshooting guidance.

## Should GenAI replace human troubleshooting?

No. It should assist the engineer, not replace validation and judgment.

## What is the best first step in Kubernetes troubleshooting?

Check events, logs, and pod status.

---

# Key Takeaway

A Markdown guide is the best format for this topic if you want something directly usable by SREs, DevOps engineers, and platform engineers in SCM or internal documentation.

An AI Skill is a strong next step only if you want a reusable assistant that follows the same troubleshooting workflow every time.

For now, this guide gives you a practical, structured, and directly usable foundation for Kubernetes troubleshooting with GenAI.
