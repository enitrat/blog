---
title: 'Cairo Coder'
summary: 'An API for Cairo code generation, pluggable into any agentic tool via MCP'
kind: 'project'
site: 'https://www.cairo-coder.com/'
order: 3
---

The principle behind Cairo Coder was, in hindsight, very simple: take a user request related to Cairo code generation, run a RAG process to find relevant code snippets, examples, and instructions related to the user's query in a database filled with Cairo-related content, inject that context into the final LLM query, and serve the response back to the user.

Cairo Coder was first built before the MCP protocol even existed. Its first shape was an OpenAI-compatible API that developers could plug into their AI coding IDE (like Cursor), and that would transparently and considerably improve the quality of Cairo-related tasks.

Eventually, Cairo Coder moved towards being an MCP server, which was easier to integrate and more versatile.
