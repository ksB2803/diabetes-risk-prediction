#!/usr/bin/env bash
# Start the Spring Boot backend + frontend on http://localhost:8080 (Linux/macOS).
# Uses the "local" profile, which enables the H2 console for localhost only.
set -euo pipefail
cd "$(dirname "$0")/../backend"

if [ -f ../.env ]; then
  set -a; . ../.env; set +a
fi

exec mvn spring-boot:run -Dspring-boot.run.profiles=local
