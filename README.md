# LLM Security Attacks and Defenses

This repository contains the codebase for an undergraduate research project focused on evaluating and mitigating privacy vulnerabilities in large language models (LLMs). The work was conducted under the mentorship of PhD student Mohamed Shaaban and Dr. Mohamed Elmahallawy at Washington State University's School of Electrical Engineering and Computer Science (WSU EECS).

## Overview

The project explores how LLMs like GPT-2 and OpenAI models can unintentionally leak sensitive information and implements both offensive and defensive techniques to study and mitigate this behavior. Two main components are included:

- **Attack Suite with GUI**  
  A `tkinter`-based graphical interface that allows researchers to interact with the OpenAI API using custom system prompts. This GUI supports execution of:
  - Extraction attacks
  - Reconstruction attacks
  - Inference attacks

- **Local Attack and Defense Scripts**  
  A separate set of scripts for running the same attacks locally on a fine-tuned GPT-2 model using:
  - Hugging Face Transformers
  - PyTorch
  - Presidio Analyzer  
  These scripts also include an implementation of **Proactive Privacy Amnesia (PPA)** as a defensive technique.

## What is Proactive Privacy Amnesia (PPA)?

PPA is a defense mechanism that proactively reduces the memorization of sensitive data in language models. Instead of redacting outputs after generation, PPA works during the training phase. The model is fine-tuned in two phases:
1. **GPT-2-forget** — The model is guided to "forget" synthetic PII it was previously exposed to.
2. **GPT-2-final** — The model is re-trained with the defense mechanism in place, significantly reducing its ability to leak sensitive information.

This method helps harden LLMs against leakage of Personally Identifiable Information (PII) by weakening associations between tokens in memory.

## Technologies Used

- Python
- Tkinter
- OpenAI API
- Transformers (Hugging Face)
- PyTorch
- Faker (for synthetic PII data)
- Microsoft Presidio

## Folder Structure
