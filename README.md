# Claude Code: A Highly Agentic Coding Assistant - Course Materials & Links

This repository contains the resources and links of the short course "Claude Code: A highly Agentic Coding Assistant".

## Course Overview

The course teaches Claude Code best practices through 3 practical examples:

- **RAG chatbot codebase** (Lessons 2-6)
- **Ecommerce data analysis** (Lesson 7) 
- **Figma design mockup implementation** (Lesson 8)

## Course Structure

- **Lesson 1**: What is Claude Code?
- **Lesson 2**: Setup & Codebase Understanding
- **Lesson 3**: Adding Features
- **Lesson 4**: Testing, Error Debugging and Code Refactoring
- **Lesson 5**: Adding Multiple Features Simultaneously
- **Lesson 6**: Exploring Github Integration & Hooks
- **Lesson 7**: Refactoring a Jupyter Notebook & Creating a Dashboard
- **Lesson 8**: Creating Web App based on a Figma Mockup

# Course Materials RAG System

A Retrieval-Augmented Generation (RAG) system designed to answer questions about course materials using semantic search and AI-powered responses.

## Overview

This application is a full-stack web application that enables users to query course materials and receive intelligent, context-aware responses. It uses ChromaDB for vector storage, Anthropic's Claude for AI generation, and provides a web interface for interaction.

## Prerequisites

- Python 3.13 or higher
- uv (Python package manager)
- An Anthropic API key (for Claude AI)
- **For Windows**: Use Git Bash to run the application commands - [Download Git for Windows](https://git-scm.com/downloads/win)

## Installation

1. **Install uv** (if not already installed)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install Python dependencies**
   ```bash
   uv sync
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```bash
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

## Running the Application

### Quick Start

Use the provided shell script:
```bash
chmod +x run.sh
./run.sh
```

### Manual Start

```bash
cd backend
uv run uvicorn app:app --reload --port 8000
```

The application will be available at:
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

## What's Included

- **Reading Notes** (`reading_notes/`) - Detailed notes for each lesson including prompts used and feature summaries
- **Lesson 7 Files** (`lesson7_files/`) - Complete ecommerce data analysis example with:
    - Jupyter notebooks (original and refactored)
    - Python modules for data loading, business metrics, and dashboard
    - Sample ecommerce datasets
- **Additional Resources** (`additional_files/`) - Supplementary materials including the visualization generated in lesson 1 and the figma binary file of the mockup used in lesson 8.
- **Course Repository Links** (`links_to_course_repos.md`) - Links to course repositories used in lessons 3-6 and lesson 8

## Resources

- [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code/overview)
- [Claude Code Common Workflows](https://docs.anthropic.com/en/docs/claude-code/common-workflows)
- [Claude Code Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)
- [Claude Code Use Cases](https://www.anthropic.com/news/how-anthropic-teams-use-claude-code)
- [Claude Code in Action - Anthropic Academy Course](https://anthropic.skilljar.com/claude-code-in-action)
