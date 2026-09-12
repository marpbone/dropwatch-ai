#!/usr/bin/env bash
# Regenerate frontend TypeScript types from the backend's OpenAPI schema.
#
# The backend's Pydantic models are the single definition of the API contract.
# Run this after changing anything in djprep/models/ so the frontend cannot
# silently drift from what the server accepts.
set -euo pipefail
cd "$(dirname "$0")/.."
python -c "
import json, sys
sys.path.insert(0, 'backend')
from djprep.api.main import app
print(json.dumps(app.openapi(), indent=2))
" > /tmp/djprep-openapi.json
npx --yes openapi-typescript /tmp/djprep-openapi.json -o frontend/src/api/schema.d.ts
echo "wrote frontend/src/api/schema.d.ts"
