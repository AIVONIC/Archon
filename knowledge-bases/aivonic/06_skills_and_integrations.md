# Aivonic - Skills & Integrations

## Overview

Aivonic agents come equipped with skills - modular capabilities that allow them to perform real actions beyond conversation. There are 31 built-in skills available, and custom skills can be created for specific business needs.

When a customer message requires an action, the agent activates the appropriate skill seamlessly within the conversation.

## Built-In Skills (31)

### Communication
- **Email Delivery** - Send emails on behalf of the business (confirmations, follow-ups, notifications)
- **SMS Sender** - Send text messages to customers
- **Slack Poster** - Post messages to Slack channels
- **Teams Notifier** - Send notifications to Microsoft Teams

### Automation
- **Browser Automation** - Interact with web pages programmatically
- **Web Scraper** - Extract data from websites
- **Form Filler** - Complete forms automatically
- **Webhook Handler** - Receive and process webhook events

### Business Operations
- **Appointment Booking** - Schedule meetings and appointments
- **Calendar Check** - Check availability and scheduling conflicts
- **CRM Integration** - Connect with customer relationship management systems
- **Order Tracking** - Look up and report order status
- **Inventory Check** - Query product availability
- **Cart Manager** - Manage shopping cart operations
- **Pricing Quotes** - Generate pricing estimates
- **Return Processor** - Handle return and exchange workflows
- **Stripe Checkout** - Process payments via Stripe

### Documents & Reports
- **PDF Generator** - Create PDF documents
- **Document Generation** - Generate formatted documents
- **Contract Generator** - Create contract documents
- **Report Builder** - Build data reports

### Data & Intelligence
- **Database Query** - Query databases for information
- **Data Transformer** - Transform and format data
- **Knowledge Retrieval** - Search the agent's knowledge base (included with every agent)
- **Sentiment Analysis** - Analyze customer sentiment
- **Web Search** - Search the web for real-time information
- **Translation** - Translate text between languages

### Utility
- **API Caller** - Make calls to external APIs
- **Screenshot Taker** - Capture screenshots of web pages
- **Reminder Scheduler** - Set and manage reminders
- **Human Escalation** - Transfer conversations to human agents with full context
- **Feedback Collection** - Gather and store customer feedback

## Skill Assignment

Skills are assigned during agent creation based on the business requirements discussed in the discovery call. Common combinations include:

**Customer Support Agent:**
knowledge_retrieval, human_escalation, email_delivery, order_tracking, feedback_collection

**Sales/Lead Generation Agent:**
knowledge_retrieval, appointment_booking, email_delivery, crm_integration, pricing_quotes

**E-Commerce Agent:**
knowledge_retrieval, order_tracking, cart_manager, inventory_check, return_processor, stripe_checkout

**Professional Services Agent:**
knowledge_retrieval, appointment_booking, email_delivery, document_generation, human_escalation

## Custom Skills

Custom skills can be created when a business has requirements not covered by the built-in library.

## Integration Points

Agents can integrate with external systems through:
- **API connections** via the API Caller skill
- **Webhook events** for real-time triggers
- **CRM systems** for customer data synchronization
- **Payment processors** (Stripe) for transactions
- **Communication platforms** (Slack, Teams, email, SMS)
- **Scheduling tools** for appointment management
- **Database connections** for direct data access
