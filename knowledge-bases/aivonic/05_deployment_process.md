# Aivonic - Deployment Process

## Onboarding Steps

### 1. Discovery Call
- Free 30-minute consultation at https://aivonic.ai/discovery
- We learn about the business, goals, customer base, and requirements
- Discuss which skills and integrations the agent needs
- Determine the right product tier

### 2. Agent Configuration
- NEXUS (our agent creation engine) builds the agent based on requirements
- AXIOM generates the system prompt - the agent's personality, knowledge, and behavior rules
- CRITIC evaluates the prompt on 8 quality dimensions (must score 8.0+ to pass)
- Up to 3 revision iterations if needed, best prompt wins
- Skills assigned based on business needs (email, scheduling, CRM, etc.)

### 3. Knowledge Base Setup
- Business-specific knowledge is ingested via Archon (our RAG system)
- Sources can include website content (crawled automatically), uploaded documents, FAQs, and product catalogs
- Knowledge is embedded and stored for semantic search
- Agent uses this knowledge to answer customer questions accurately

### 4. Pre-Deployment Evaluation

Every agent passes a mandatory 4-phase quality evaluation before going live:

**Phase 1 - Infrastructure**
- Configuration validation
- API key verification
- LLM connectivity testing
- All technical dependencies confirmed working

**Phase 2 - Functional**
- Automated test cases covering greeting, skills, and guardrails
- Verifies the agent can perform its assigned tasks correctly
- Tests edge cases and error handling

**Phase 3 - Quality**
- LLM-as-Judge scoring on relevance, accuracy, and helpfulness
- Evaluates response quality across a range of scenarios
- Ensures the agent meets the standard for its certification level

**Phase 4 - Security**
- Prompt injection resistance testing
- Jailbreak protection verification
- Ensures the agent stays within its defined boundaries

### 5. Certification

Based on evaluation scores, agents receive a certification level:

| Level | Badge | Quality Score | Functional Rate |
|-------|-------|---------------|-----------------|
| Standard | Check mark | 7.0+ out of 10 | 90%+ |
| Premium | Star | 8.0+ out of 10 | 95%+ |
| Elite | Trophy | 8.5+ out of 10 | 100% |

### 6. Deployment
- Chat widget is installed on the client's website (single script tag)
- Widget is customizable: colors, logo, title, theme (dark/light)
- Agent goes live and starts serving customers
- Client gets access to the dashboard for analytics, conversation history, and settings

### 7. Ongoing Management
- We handle monitoring, updates, and optimization
- Support available per the client's plan (Standard or Premium)
- Knowledge base can be updated as the business evolves
- Skills and integrations can be added over time

## Timeline

- Agent can be production-ready same day for straightforward deployments
- More complex setups (multiple integrations, large knowledge bases) may take a few days
- The client's total time investment is approximately 1 hour (discovery call + onboarding information)

## Client Dashboard

Once deployed, clients have access to:
- **Dashboard** - Overview of agent activity
- **Conversations** - Full chat history with customers
- **Analytics** - Usage statistics and performance metrics
- **Settings** - Widget customization, voice settings
- **Integrations** - Integration marketplace
- **Billing** - Subscription management
- **Live Support** - Real-time support monitoring
