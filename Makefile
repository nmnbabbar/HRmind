# =============================================================================
# Makefile — HrMind developer shortcuts
# =============================================================================

.DEFAULT_GOAL := help
.PHONY: help build up down restart logs shell test test-phase1 test-phase2 \
        ingest seed-db eval clean ps

# ── Meta ──────────────────────────────────────────────────────────────────────

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Docker lifecycle ──────────────────────────────────────────────────────────

build: ## Build all Docker images (no cache bust)
	docker compose build

build-nc: ## Build all Docker images (force no cache)
	docker compose build --no-cache

up: ## Start all services in detached mode
	docker compose up -d

down: ## Stop all services
	docker compose down

restart: ## Restart backend only (fast code reload)
	docker compose restart backend

logs: ## Follow backend logs (Ctrl+C to stop)
	docker compose logs -f backend

logs-all: ## Follow all service logs
	docker compose logs -f

ps: ## Show service status
	docker compose ps

# ── Development ───────────────────────────────────────────────────────────────

shell: ## Open a bash shell inside the backend container
	docker compose exec backend bash

shell-chroma: ## Open a shell in the chromadb container
	docker compose exec chromadb sh

# ── Testing ───────────────────────────────────────────────────────────────────

test: ## Run all tests inside backend container
	docker compose exec backend pytest tests/ -v --tb=short

test-phase1: ## Run Phase 1 foundation tests only
	docker compose exec backend pytest tests/test_phase1.py -v

test-phase2: ## Run Phase 2 RAG agent tests
	docker compose exec backend pytest tests/test_rag_agent.py -v

test-phase3: ## Run Phase 3 SQL agent tests
	docker compose exec backend pytest tests/test_sql_agent.py -v

test-phase4: ## Run Phase 4 Doc Parser tests
	docker compose exec backend pytest tests/test_doc_parser.py -v

test-phase5: ## Run Phase 5 Orchestration tests
	docker compose exec backend pytest tests/test_orchestration.py -v

test-api: ## Run Phase 6 API tests
	docker compose exec backend pytest tests/test_api.py -v

test-e2e: ## Run Phase 8 end-to-end tests
	docker compose exec backend pytest tests/test_e2e.py -v

# ── Data operations ───────────────────────────────────────────────────────────

ingest: ## Ingest HR documents into ChromaDB (Phase 2+)
	docker compose exec backend python -m backend.agents.rag_agent.ingestion

seed-db: ## Seed SQLite HR database with sample data (Phase 3+)
	docker compose exec backend python -m backend.agents.sql_agent.seed.seed_db

eval: ## Run RAGAS evaluation on RAG agent (Phase 2+)
	docker compose exec backend python -m backend.agents.rag_agent.evals.eval_runner

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean: ## Remove containers, volumes, and orphan services
	docker compose down -v --remove-orphans

clean-chroma: ## Clear only the ChromaDB volume (force re-ingestion)
	docker compose down chromadb
	docker volume rm hrmind_chroma_data 2>/dev/null || true
	docker compose up -d chromadb
