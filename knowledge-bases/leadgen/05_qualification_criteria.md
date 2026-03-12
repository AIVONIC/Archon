# Lead Qualification & Scoring Criteria

## Qualification Score (0-100)

The leadgen agent scores each discovered business on these factors. Businesses scoring 60+ are qualified for outreach.

### Scoring Factors

| Factor | Weight | Scoring Logic |
|--------|--------|---------------|
| Has website | 20 pts | URL present and accessible |
| Has email | 20 pts | Business email found (not personal) |
| Google rating | 15 pts | 4.0+ = 15, 3.5-3.9 = 10, 3.0-3.4 = 5, <3.0 = 0 |
| Review count | 15 pts | 50+ = 15, 20-49 = 10, 5-19 = 5, <5 = 0 |
| Industry fit | 20 pts | Target industry = 20, adjacent = 10, poor fit = 0 |
| Employee count | 10 pts | 5-50 = 10, 51-200 = 7, 1-4 = 3, 200+ = 2 |

### Qualification Thresholds

| Score | Classification | Action |
|-------|---------------|--------|
| 80-100 | Hot lead | Priority outreach |
| 60-79 | Qualified lead | Standard outreach |
| 40-59 | Marginal lead | Hold for review |
| 0-39 | Disqualified | Skip |

## Target Industries (Priority Order)

### Tier 1 — Best Fit (industry_score = 20)
- Marketing agencies
- Digital agencies
- Creative agencies
- PR firms
- Real estate agencies/brokerages
- SaaS companies
- Software companies
- IT consulting firms

### Tier 2 — Good Fit (industry_score = 15)
- Architecture firms
- Design studios
- Management consulting
- E-commerce stores
- Professional services (general)
- Accounting firms
- Law firms

### Tier 3 — Moderate Fit (industry_score = 10)
- Restaurants / hospitality (chain/multi-location only)
- Healthcare clinics (non-HIPAA sensitive)
- Recruitment agencies
- Insurance agencies
- Training/education companies

### Tier 4 — Poor Fit (industry_score = 0)
- Solo practitioners / freelancers
- Government entities
- Non-profits (budget constraints)
- Heavy industry / manufacturing
- Retail (single location, <5 employees)

## Target Company Size

**Sweet spot: 5-50 employees**
- Large enough to have real customer inquiry volume
- Small enough that $5K-$8K setup is a reasonable investment
- Don't have dedicated support teams yet
- Decision-making is faster (often founder or VP)

**Also viable: 51-200 employees**
- More budget available
- Longer sales cycle
- May need multi-agent setup

**Avoid for cold outreach:**
- 1-4 employees (usually can't afford $5K+ setup)
- 200+ employees (enterprise sales process needed, not cold email)

## Target Regions (Priority Order)

1. **United States** — largest market, English-speaking, highest willingness to pay
2. **United Kingdom** — English-speaking, strong services sector
3. **Sweden** — home market, Aivonic is Swedish
4. **Nordics** (Norway, Denmark, Finland) — similar market to Sweden
5. **EU** (Germany, Netherlands, France) — mature markets, GDPR-aware

## Disqualification Criteria (Auto-Skip)

Skip any lead that matches:
- No website found
- No business email found (only personal @gmail, @yahoo, etc.)
- Google rating below 2.5 (reputation issues)
- Permanently closed
- Already in suppression list
- Contacted within last 90 days (cooldown)
- Industry is in Tier 4 (poor fit)
- Clearly a franchise/chain with central management

## Research Priorities

When crawling a qualified lead's website, extract:
1. **Services offered** — what they actually do (not just industry category)
2. **Key people** — founders, partners, leadership team names
3. **Recent news/updates** — blog posts, press releases, awards
4. **Pain point signals** — hiring pages (growing), service expansion, new locations
5. **Technology signals** — existing chatbot? Contact form only? Phone-only?
6. **Content quality** — well-maintained site vs. outdated (indicates investment capacity)

## Email Personalization Requirements

**Minimum personalization for sending:**
- Reference at least ONE specific thing about the business (not just industry)
- Mention their location or market
- Connect to a relevant pain point for their size/industry

**Examples of "specific enough" personalization:**
- "I noticed your Austin office recently expanded to commercial real estate"
- "Your blog post on sustainable design got me thinking"
- "With 4.8 stars and 200+ reviews, [Company] clearly delivers"

**NOT specific enough (would be held for review):**
- "I noticed you're a marketing agency" (too generic)
- "Your business could benefit from AI" (generic)
- "Companies in your industry are using AI" (no personalization)
