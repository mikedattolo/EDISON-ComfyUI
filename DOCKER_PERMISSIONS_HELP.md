# Docker Permissions & Branding Error Fix

If you encounter: `Permission denied: '/opt/edison/outputs/clients'` when creating a new client in branding, it's a volume mount permission issue in Docker.

## Quick Fix

The app now automatically attempts to initialize these directories on startup. Check the Docker logs for any warnings:

```bash
docker-compose logs edison-core | grep -i "integration\|permission"
```

If you see warnings, try these fixes:

## Fix #1: Ensure Output Directories Exist (Recommended)

```bash
# Create the required directories with proper permissions
mkdir -p outputs/clients/{images,videos,files}
mkdir -p config/integrations
mkdir -p outputs outputs/clients config/integrations

# Make sure they're writable (Linux/Mac)
chmod -R 755 outputs config
```

## Fix #2: Restart Docker with Fresh Volumes

```bash
# Stop containers and remove volumes
docker-compose down -v

# Recreate with fresh volumes
docker-compose up -d

# Check logs
docker-compose logs -f edison-core
```

## Fix #3: Run Init Endpoint on First Start

Once the container is running, call the initialization endpoint to verify/fix permissions:

```bash
curl -X POST http://localhost:8811/api/system/initialize
```

You should see a response like:
```json
{
  "ok": true,
  "message": "System initialized successfully",
  "directories": {
    "integrations": "/path/to/config/integrations",
    "branding": "/path/to/outputs/clients",
    "config": "/path/to/config"
  },
  "writable": true
}
```

## Fix #4: Docker Compose Permission Settings (Advanced)

If the above doesn't work, update your `docker-compose.yml` to handle permissions:

```yaml
services:
  edison-core:
    # ... other config ...
    volumes:
      - ./models:/opt/edison/models
      - ./data:/opt/edison/data
      - ./outputs:/opt/edison/outputs
      - ./config:/opt/edison/config
      - ./:/opt/edison
    # Optional: run as specific user (adjust uid:gid as needed)
    # user: "1000:1000"
```

Then ensure host directories have correct permissions before starting:

```bash
# Set permissions for mounted volumes (Linux)
chmod 777 outputs config
```

## Fix #5: Use Named Volumes (For Production)

For better permission handling in production, use named volumes:

```yaml
services:
  edison-core:
    volumes:
      - edison-models:/opt/edison/models
      - edison-data:/opt/edison/data
      - edison-outputs:/opt/edison/outputs
      - edison-config:/opt/edison/config

volumes:
  edison-models:
  edison-data:
  edison-outputs:
  edison-config:
```

## Windows WSL2 Permissions

If running on Windows with WSL2:

1. Ensure the project folder is on the WSL filesystem (not /mnt/c):
```bash
wsl --list --verbose  # Check your WSL version
```

2. Rebuild the volumes:
```bash
docker-compose down -v
docker-compose up -d
```

## Verify It's Fixed

Try creating a new client in the branding page:
1. Go to `/branding` page
2. Click "Create New Client"
3. Enter a name like "Test Client"
4. Should succeed without permission errors

If you still have issues, check:
- Docker logs: `docker-compose logs edison-core`
- File permissions on host: `ls -la outputs/ config/`
- Docker user running the container: `docker ps` then `docker inspect <container_id>`

## What's Happening

The EDISON branding system stores client assets (images, videos, files) in `outputs/clients/`. When you create a new client:

1. App calls `_ensure_integrations_dir()` to verify parent directories exist
2. Creates client-specific subdirectories: `outputs/clients/{client-slug}/{images,videos,files}`
3. Stores metadata in `config/integrations/branding.json`

If the parent directory doesn't exist or isn't writable, the mkdir() call fails with Permission Denied.

The fix ensures these directories are:
- Created at app startup
- Created recursively before client creation
- Wrapped with helpful error messages pointing to Docker volume issues if they fail
