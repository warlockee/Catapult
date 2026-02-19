# Fix for Persistent 401 Unauthorized Errors

## Problem
The application was returning `401 Unauthorized` errors for all API requests, with the error message "Invalid API key". This persisted even after verifying the API key in the frontend code.

## Root Cause
1.  **Backend Salt Mismatch**: The API key hash stored in the database likely did not match the hash generated with the current `API_KEY_SALT`. This can happen if the salt environment variable changes or if the database state is inconsistent with the configuration.
2.  **Frontend Caching**: The frontend application cached the API key in `localStorage`. Even after fixing the backend key, the frontend continued to send the old, invalid key.

## Solution

### Backend Fix
Modified `backend/scripts/create_api_key.py` to add a `--reset` flag. This allows regenerating and updating an existing API key's hash in the database without needing to manually delete the record first.

**Command to reset key:**
```bash
docker compose exec backend python scripts/create_api_key.py --name "admin" --reset
```

### Frontend Fix
Updated `frontend/src/lib/api.ts` to:
1.  Use the new, valid API key as the default `DEMO_API_KEY`.
2.  Add logic to detect the specific old invalid key in `localStorage` and automatically replace it with the new valid key.

### Deployment
Redeployed the application using `./deploy.sh` to apply both backend and frontend changes.

## Verification
- **Backend**: Verified using `curl` with the new API key.
- **Frontend**: Verified by reloading the page; the auto-correction logic updated the local storage, and API requests succeeded.
