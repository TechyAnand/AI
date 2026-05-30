# MCP Complete Learning Guide
## Author: Kartheek Anand

## Purpose

This guide introduces the Model Context Protocol (MCP) and shows how to build a very small local MCP server for learning and practice.

It is designed for:

- Beginners
- DevOps Engineers
- DevSecOps Engineers
- AI Engineers
- Automation Engineers
- Interview Preparation

By the end of this guide, you will understand:

- What MCP is
- Why MCP matters
- Core MCP concepts
- MCP architecture
- MCP tools, resources, and prompts
- How a local MCP server works
- How to run a simple MCP example locally
- Best practices
- Common interview questions

---

# What is MCP?

MCP stands for **Model Context Protocol**.

It is a standard way for AI applications to connect to external tools, services, and data sources.

In simple words:

    MCP helps AI models talk to external systems in a structured way.

Examples of what MCP can connect to:

- Files
- Databases
- APIs
- Internal tools
- Local utilities
- Knowledge sources

---

# Why MCP?

Without MCP, every AI application may need a custom integration for every tool or data source.

That creates:

- duplicated effort
- inconsistent integrations
- harder maintenance
- poor reuse

With MCP:

- integrations become standardized
- tools can be reused across clients
- AI applications can connect to context more cleanly
- development becomes easier and more modular

---

# MCP in Simple Terms

Think of MCP as a USB-C port for AI tools.

Just like USB-C provides a common interface for many devices, MCP provides a common interface for AI applications to connect to tools and context.

---

# Core MCP Concepts

## 1. Host

The AI application that wants to use tools or context.

Examples:

- Chat application
- IDE assistant
- AI agent
- Desktop AI client

## 2. MCP Server

A service that exposes tools, resources, or prompts.

It is the bridge between the host and external capabilities.

## 3. Tools

Actions the AI can call.

Examples:

- calculate
- read file
- query data
- fetch system info

## 4. Resources

Read-only or file-like context exposed to the client.

Examples:

- configuration data
- documents
- notes
- API response content

## 5. Prompts

Reusable prompt templates shared by the server.

These help guide AI behavior for specific tasks.

---

# MCP Architecture

    Host / AI Client
            |
            v
        MCP Server
        /   |   \\
       /    |    \\
   Tools  Resources Prompts
       \    |    /
            v
     External Systems / Data

---

# Why MCP is Useful for GenAI

MCP helps generative AI systems:

- access real-world data
- use external tools safely
- reuse integrations across apps
- reduce custom code
- keep AI systems modular
- make assistants more useful in practice

---

# MCP vs Traditional API Integration

## Traditional approach

Every app talks to every system in a custom way.

## MCP approach

AI applications talk to MCP servers using a standard protocol.

This gives:

- better reuse
- consistent behavior
- easier maintenance
- easier scaling of AI tool ecosystems

---

# Local MCP Example Overview

In this guide, we will build a tiny local MCP server that exposes:

- a calculator tool
- a greeting resource
- a reusable prompt

This is a beginner-friendly example that can run on your machine.

---

# Project Structure

    mcp-local-demo/
    └── server.py

You can keep this simple in the beginning.

---

# Simple Local MCP Server Example

Save the following as `server.py`.

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Local Demo Server", json_response=True)

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Return a simple greeting"""
    return f"Hello, {name}! Welcome to MCP."

@mcp.prompt()
def write_summary(topic: str) -> str:
    """Generate a simple prompt template"""
    return f"Write a short summary about {topic} for a beginner."

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

---

# What This Example Does

## Tool: add

Adds two numbers.

Example:

    add(2, 3)

Result:

    5

## Tool: multiply

Multiplies two numbers.

Example:

    multiply(4, 5)

Result:

    20

## Resource: greeting://{name}

Returns a personalized message.

Example:

    greeting://Kartheek

Result:

    Hello, Kartheek! Welcome to MCP.

## Prompt: write_summary

Creates a reusable prompt template.

Example input:

    topic = Kubernetes

Generated prompt:

    Write a short summary about Kubernetes for a beginner.

---

# How to Run the Local MCP Server

## Prerequisites

Install:

- Python 3.10 or newer
- `uv` or another Python package runner

---

## Run the server

Using `uv`:

    uv run --with mcp server.py

This starts the MCP server locally.

---

# How to Test the Server

You can test the server using the MCP Inspector.

Start the inspector:

    npx -y @modelcontextprotocol/inspector

Then connect to:

    http://localhost:8000/mcp

From there, you can call the tools and inspect the resources.

---

# What Happens Behind the Scenes

When a client uses MCP:

1. The client sends a request.
2. The MCP server receives the request.
3. The server runs the requested tool or returns the requested resource.
4. The result is sent back to the client.
5. The AI uses the result to form a response.

---

# MCP Use Cases

MCP is useful for:

- AI assistants
- Enterprise knowledge systems
- IDE integrations
- Developer tooling
- File-based assistants
- API-based assistants
- Automation workflows

---

# How to Use MCP in Real Projects

A common workflow is:

    AI Client
        |
        v
    MCP Server
        |
        v
    Files / APIs / Databases / Tools

Examples:

- AI assistant reads project docs
- AI assistant queries internal APIs
- AI assistant fetches configuration details
- AI assistant triggers allowed automation steps

---

# Security Considerations

When using MCP in real systems:

- expose only approved tools
- restrict access to sensitive data
- validate inputs carefully
- use authentication where needed
- log important actions
- avoid overexposing internal systems
- treat tool execution as privileged

Security is especially important when MCP is used in enterprise environments.

---

# Best Practices

1. Keep tools small and focused
2. Return predictable outputs
3. Validate input parameters
4. Avoid exposing unnecessary secrets
5. Separate read-only and write actions
6. Use clear names for tools and resources
7. Keep prompts reusable
8. Test locally before production
9. Add logging for debugging
10. Document the server clearly in README files

---

# Common Mistakes

## Too many responsibilities in one tool

Bad:

    a tool that fetches data, transforms it, writes files, and sends emails

Better:

    separate tools for each task

---

## Exposing sensitive data

Do not expose secrets unless absolutely required.

---

## Unclear tool names

Bad:

    doStuff

Better:

    add_numbers
    fetch_user_profile
    list_repositories

---

## Poor input validation

Always check types and required parameters.

---

# MCP and DevOps

MCP can be useful in DevOps workflows such as:

- reading logs
- querying monitoring systems
- reading deployment metadata
- fetching cloud inventory
- helping with documentation
- assisting with troubleshooting

This makes it useful for AI-assisted operations and productivity tooling.

---

# MCP and DevSecOps

MCP can support DevSecOps workflows such as:

- security report retrieval
- vulnerability lookup
- configuration review
- compliance checks
- audit log access
- policy guidance

The key is to expose only safe, approved capabilities.

---

# Interview Questions

## What is MCP?

MCP is the Model Context Protocol, a standard way for AI applications to connect to external tools and data sources.

## Why is MCP useful?

It standardizes integrations and makes AI systems easier to extend.

## What are MCP tools?

Actions that the client can request the server to execute.

## What are MCP resources?

Read-only or file-like context exposed by the server.

## What are MCP prompts?

Reusable prompt templates exposed by the server.

## Can MCP be used locally?

Yes. MCP servers can run locally for development and testing.

---

# Quick Revision Cheat Sheet

MCP:

    Model Context Protocol

Core Parts:

    Host
    Server
    Tools
    Resources
    Prompts

Local Run Example:

    uv run --with mcp server.py

Inspector:

    npx -y @modelcontextprotocol/inspector

Simple Example Tools:

    add
    multiply

Simple Example Resource:

    greeting://{name}

---

# Key Takeaway

MCP gives AI applications a standard way to work with external tools and context.

A simple local MCP server is a great way to learn the concept before moving to production use cases.

Mastering MCP helps with:

- AI assistant integrations
- automation
- enterprise tool access
- developer productivity
- secure AI workflows
