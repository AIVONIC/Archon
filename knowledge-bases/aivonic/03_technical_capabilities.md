# Aivonic - Technical Capabilities

## Platform Architecture

The Aivonic platform is a full-stack AI agent system with multiple specialized components working together to deliver intelligent, reliable, and secure agents.

### Core Systems

**NEXUS** - Agent Creation Engine
- Creates agents via a structured 3-step flow
- Uses AXIOM (prompt engineer) and CRITIC (quality evaluator) sub-agents for prompt engineering
- AXIOM generates system prompts, CRITIC scores on 8 dimensions
- Prompts must score 8.0+ average to pass (max 3 revision iterations)
- 31 built-in skills available for assignment
- Agents can be created for client deployment or platform-only use

**ATLAS** - 5-Agent Reasoning Pipeline
- Advanced reasoning system with five specialized agents: Analyzer, Strategist, Critic, Synthesizer, Reflector
- Handles complex queries that require multi-step reasoning
- ATLAS is a pure reasoning system with no skills - it focuses entirely on deep analysis

**Archon** - Knowledge Base System
- RAG (Retrieval-Augmented Generation) with Supabase pgvector
- Allows agents to access business-specific knowledge
- Web crawling and document upload for knowledge ingestion
- Semantic search over embedded documents

## Multilingual Support

- Automatic language detection on every message
- Agent responds in the same language the customer uses
- No configuration required - works out of the box
- Supported via ElevenLabs multilingual voice model for TTS

## Voice Capabilities

Three voice provider options:
1. **ElevenLabs** - High-quality text-to-speech, multilingual support
2. **Hume AI** - Emotionally expressive text-to-speech
3. **Browser Speech API** - Voice input via Web Speech API (free, built-in)

Voice behavior:
- TTS only triggers on voice input (not on typed messages)
- Widget passes input mode metadata so backend knows when to generate audio
- Clients can configure voice settings through the dashboard

## Infinite Memory

- Every customer interaction is remembered permanently
- 50M+ message capacity
- Customers never have to repeat themselves across sessions
- Enables truly personalized experiences that improve over time
- Standard chatbots forget after roughly 20 messages or when the session ends

## Skills System

31 built-in skills across multiple categories:
- Communication: email delivery, SMS sender, Slack poster, Teams notifier
- Automation: browser automation, web scraper, form filler, webhook handler
- Business: appointment booking, CRM integration, order tracking, inventory check
- Documents: PDF generator, document generation, contract generator, report builder
- Data: database query, data transformer, knowledge retrieval, sentiment analysis
- Utility: web search, API caller, screenshot taker, translation, reminder scheduler

Skills are routed via regex and semantic matching. NEXUS can also create new custom skills dynamically.

## Security

**Crust Security Gateway**
- Data Loss Prevention (DLP)
- Prompt injection protection
- Jailbreak resistance testing during pre-deployment evaluation

**Pre-Deployment Security Phase**
- Automated prompt injection resistance testing
- Jailbreak protection verification
- Part of the mandatory 4-phase evaluation every agent undergoes

## LLM Support

- **OpenAI models** (GPT-5.2 and newer)
- **Anthropic Claude models**
- Model selection based on agent requirements and use case

## Chat Widget

- Embeddable on any website
- Voice input and output support
- File upload capability
- Dark and light theme options
- Customizable branding (colors, logo, title)
- Real-time messaging via WebSocket
