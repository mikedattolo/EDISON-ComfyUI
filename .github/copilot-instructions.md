# EDISON Copilot Workspace Instructions

# Objective
Transform the existing EDISON AI assistant into a full-service marketing, branding, fabrication, and content-production platform without losing its identity as an AI assistant similar to ChatGPT.

EDISON must still function as a conversational AI first:
- accept normal user chat input
- answer questions naturally
- maintain context across a conversation
- reason through requests
- help with coding, planning, research, and general tasks
- preserve all current chat modes and existing features

At the same time, EDISON should become an operational business tool that can actively help run branding, marketing, fabrication, video, and client-service workflows.

The final system must keep all current functionality intact while adding business-oriented capabilities on top of the existing AI core.

---

# Critical Requirements

## 1) Preserve EDISON as a ChatGPT-like AI Assistant
Do not turn EDISON into a narrow dashboard-only app.

The system must still:
- work as a general conversational AI assistant
- accept freeform user input in chat
- support follow-up questions and ongoing conversations
- understand user intent automatically
- retain all current modes and routing behavior unless a change is necessary for stability
- keep existing deep search, reasoning, code, agent, work, swarm, branding, video, printing, and image-generation capabilities
- preserve the current reverse-proxy architecture and front-end chat experience unless improvements are clearly beneficial

The assistant should remain the central interface. All new branding, marketing, fabrication, and project-management features should be accessible either:
1. directly through natural language in chat, or
2. through dedicated UI pages that are also tied back into the assistant

Examples:
- “Make a branding package for Adoro Pizza”
- “Generate 5 logo concepts and a slogan”
- “Turn this logo into a printable keychain STL”
- “Make a promo video shot list for this restaurant”
- “Create a client folder and organize all related assets”
- “Review my current code and tell me what’s broken in branding mode”
- “Post this design to Instagram next Friday at 2 PM”
- “Use my current project files to make a social campaign”

---

## 2) Keep All Existing Features
Do not remove, weaken, or break any existing functionality.

Must preserve:
- general chat
- instant/chat/reasoning/thinking/code/agent/swarm/work modes
- image generation
- gallery
- file manager
- branding storage
- video editing
- connectors
- 3D printing
- deep search
- prompt systems
- model hot-swap
- settings sync
- system diagnostics/readiness
- current API endpoints unless new versions are needed
- existing JSON storage patterns where appropriate
- OpenAI-like behavior where already implemented

Any refactor must be backward-compatible where possible.

---

## 3) EDISON Must Be Aware of Its Own System and Codebase
EDISON needs stronger self-awareness of its own architecture, code, features, tools, and system state.

Implement system/code awareness so the assistant can:
- understand its own modules, endpoints, pages, workflows, and configuration
- reason about its own codebase when the user asks for improvements or bug fixes
- inspect relevant files safely
- explain how its own features work
- troubleshoot its own services
- understand branding/video/printing/connectors/project state
- reference internal capabilities before answering users
- distinguish between what it can do now vs. what requires setup

Examples:
- “Why is branding client creation failing?”
- “What routes handle printer registration?”
- “Which page controls video editing?”
- “What part of the code manages client storage?”
- “Can you inspect your own branding workflow and suggest improvements?”
- “Which endpoints exist for printing and what’s missing?”

Add an internal capability layer or knowledge index that maps:
- front-end pages
- API routes
- core service modules
- config files
- storage files
- integrations
- printer/video/branding services
- model modes and tools

This should allow the assistant to answer questions about its own implementation and use that understanding while executing tasks.

---

## 4) EDISON Must Be Able to Execute Business-Oriented Commands
Add and improve action execution so EDISON can perform commands related to:
- branding
- marketing
- promotional content
- fabrication
- 3D printing
- asset organization
- project workflows
- client management
- video production
- social posting
- design preparation

These actions should be accessible through natural language chat and/or dedicated UIs.

Examples of executable intents:
- create client folder
- build a project workspace
- generate logo concepts
- generate marketing copy
- render social-media templates
- organize uploaded assets into a client/project
- create fabrication-ready STL files
- slice and queue prints
- generate signage mockups
- create product promo plans
- generate storyboards and shot lists
- batch prepare campaign deliverables
- schedule social posts
- produce printable/exportable files

Use structured tools/endpoints behind the scenes, but make the user experience feel like they are just talking to the assistant.

---

# Implementation Tasks

## A. Strengthen the Core AI Assistant Layer
1. Preserve and improve conversational AI behavior across all modes.
2. Ensure natural-language requests can trigger business workflows without requiring the user to navigate manually.
3. Improve intent routing so EDISON can distinguish:
   - general conversation
   - code help
   - self-debugging/system questions
   - branding tasks
   - marketing tasks
   - fabrication/printing tasks
   - video tasks
   - project management tasks
4. Add a unified action planner so the assistant can decompose requests into:
   - conversation response
   - required tool calls
   - file operations
   - UI updates
   - project/task updates
5. Keep the assistant responsive even when actions are available; it should still explain, guide, and confirm outcomes clearly.

---

## B. Add Internal System Awareness / Self-Inspection
1. Create an internal registry or capability map describing:
   - all major pages in `/web`
   - all major backend endpoints
   - JSON storage files and schemas
   - environment variables
   - available tools/modes
   - connectors
   - printing services
   - branding services
   - video services
2. Add backend endpoints or internal helper functions that let the assistant inspect:
   - available modules
   - route definitions
   - system readiness
   - connector availability
   - branding/printer/project records
   - relevant config state
3. Add safe code-inspection abilities:
   - read project files
   - search for relevant functions/classes/routes
   - summarize how components work
   - suggest code changes based on actual files
4. Prevent unsafe self-modification by default. Any code-editing capability must be controlled, logged, and reversible.

---

## C. Client & Project Management
1. Extend the branding/client model to include:
   - business name
   - contact person
   - email
   - phone
   - website
   - industry
   - notes
   - tags
2. Add a project model with:
   - client reference
   - project name
   - description
   - service type(s): branding, printing, video, marketing, mixed
   - due date
   - status
   - assets
   - tasks
   - approvals
   - deliverables
3. Add CRUD APIs under `/api/projects`.
4. Build a project dashboard UI.
5. Link projects to branding assets, print jobs, video jobs, and copywriting outputs.

---

## D. Brand Design & Marketing Creation
1. Add a dedicated brand designer workflow.
2. Create endpoints for:
   - logo concept generation
   - moodboard generation
   - palette recommendations
   - typography suggestions
   - slogan/tagline generation
   - brand voice generation
   - style guide generation
3. Save outputs into client/project folders automatically.
4. Add approval/revision states for generated branding assets.
5. Add reusable template rendering for:
   - business cards
   - flyers
   - social posts
   - posters
   - banner ads
   - menus
   - signage mockups
6. Add copywriting endpoints for:
   - ad copy
   - social captions
   - email campaigns
   - business descriptions
   - product copy
   - website hero text
7. Make these functions callable from chat with plain English prompts.

---

## E. Fabrication & 3D Printing Expansion
1. Preserve current printer registration, discovery, and slicing workflows.
2. Add fabrication-focused tools:
   - logo/image to SVG cleanup
   - SVG/PNG to extruded STL
   - embossed/debossed signage generator
   - keychain generator
   - plaque generator
   - nameplate generator
   - promotional item generator
3. Add job tracking for:
   - material
   - color
   - quantity
   - estimated time
   - estimated cost
   - machine used
   - print status
4. Add support for fabrication order management, not just printer control.
5. Build printable-product presets for business promo use cases.

---

## F. Video & Content Production
1. Preserve existing video editing features.
2. Expand the video system to support:
   - project-based clip organization
   - storyboard generation
   - shot list generation
   - script generation
   - social video planning
   - promo campaign planning
3. Add optional camera-ingest support for devices like Canon 90D where technically feasible.
4. Improve the video UI into a production workspace rather than a single-operation tool panel.
5. Save video outputs and planning docs into the linked client/project.

---

## G. Social / Connector Workflows
1. Preserve current connector architecture.
2. Add business-oriented connector types:
   - Instagram
   - Facebook
   - TikTok
   - LinkedIn
   - Google Business Profile
   - email marketing tools if appropriate
3. Support:
   - draft post creation
   - scheduled posting
   - post status tracking
   - caption/image pairing
   - campaign linkage to projects
4. Make connector actions available through natural language.

---

## H. Workflow Orchestration
1. Add a unified workflow engine that can coordinate:
   - chat reasoning
   - branding generation
   - marketing copy
   - printing tasks
   - video tasks
   - file movement
   - project/task updates
2. When a user gives a complex request, EDISON should:
   - understand the goal
   - generate a plan
   - execute the appropriate steps
   - report progress
   - store outputs in the right place
3. Use existing work/swarm/agent patterns where possible instead of rebuilding from scratch.

---

## I. Authentication, Roles, and Business Readiness
1. Add optional authentication and role support for future multi-user business use.
2. Roles may include:
   - admin
   - designer
   - fabricator
   - editor
   - client-viewer
3. Protect destructive or sensitive actions.
4. Keep local single-user mode supported.

---

## J. Documentation & Help
1. Create documentation for all new workflows.
2. Add internal developer docs for architecture changes.
3. Add user-facing help inside the app.
4. Ensure the assistant can answer questions using this documentation.

---

# Technical Guidance

## Backend
- Continue using FastAPI unless a very strong reason exists to change.
- Keep the current API style and error-handling conventions.
- Prefer modular additions instead of turning `app.py` into a monolith.
- Split new business logic into modules such as:
  - `projects.py`
  - `branding_ops.py`
  - `marketing_ops.py`
  - `fabrication_ops.py`
  - `video_ops.py`
  - `system_awareness.py`
  - `workflow_engine.py`

## Frontend
- Keep current UI working.
- Add new pages/components cleanly.
- Tie business pages back into the main assistant instead of creating isolated tools.
- The main chat should be able to launch, monitor, and explain actions happening in other modules.

## Storage
- Preserve existing JSON-based local storage where practical.
- Add schemas for:
  - projects
  - tasks
  - print jobs
  - scheduled posts
  - approvals
- Make storage migration safe and backward-compatible.

## Safety / Stability
- Do not delete existing endpoints unless replacements and compatibility layers are provided.
- Log all action executions.
- Validate file paths and permissions.
- Preserve current offline/local-first philosophy wherever possible.

---

# Acceptance Criteria

The implementation is successful only if all of the following are true:

1. EDISON still feels like a conversational AI assistant similar to ChatGPT.
2. Existing chat and tool features continue to work.
3. The assistant can understand and discuss its own codebase and system structure.
4. The assistant can execute business-related branding/marketing/fabrication commands from chat.
5. Dedicated UI pages exist for deeper workflows, but chat remains the main interface.
6. Client/project management is added and linked to branding, video, and fabrication work.
7. Outputs are automatically organized into the right folders/projects.
8. New functionality is modular, documented, and tested.
9. Existing users are not broken by the changes.

---

# Deliverables
Provide:
- updated backend code
- updated frontend pages/components
- new modules for project/business workflows
- migration-safe storage updates
- tests for new APIs and workflows
- documentation for both developers and end users

Focus on making EDISON a true AI-powered business operating system for branding, marketing, fabrication, and promotional production while still preserving and strengthening its core identity as a smart, conversational, code-aware AI assistant.