# OAuth2 Setup Guide for EDISON Connectors

## Overview

EDISON now supports automatic OAuth2 authentication for popular services, eliminating the need to manually copy and paste API keys. Users simply click "Connect with [Provider]" and follow the OAuth flow.

## Supported Providers

The following providers support OAuth2 in EDISON:

- **GitHub** - Access repositories, gists, and user data
- **Gmail** - Read emails and send messages
- **Google Drive** - Access files and folders
- **Slack** - Interact with Slack workspaces
- **Notion** - Access Notion databases and pages
- **Dropbox** - Access files and folders
- **Discord** - Interact with Discord servers

## How It Works

### User Flow
1. User clicks "Connect with [Provider]" button in EDISON UI
2. Frontend calls `POST /integrations/connectors/oauth-start/{provider}`
3. Backend generates OAuth URL and returns it
4. User is redirected to provider's login/authorization page
5. User grants EDISON permission
6. Provider redirects to `POST /integrations/connectors/oauth-callback`
7. Backend exchanges authorization code for access token
8. Token is securely stored in `config/integrations/connectors.json`
9. User is redirected back to EDISON UI with success message

### Backend Flow
```
User → oauth-start → Generate State Token → Build Auth URL
                                                  ↓
                                    Provider OAuth Page
                                                  ↓
                                           User Authorizes
                                                  ↓
oauth-callback ← Authorization Code ← Provider Redirect
     ↓
Exchange Code for Token
     ↓
Store Token in Connectors DB
     ↓
Return Success
```

## Environment Configuration

### Required Environment Variables

Each OAuth provider requires CLIENT_ID and CLIENT_SECRET credentials. Set these in your environment:

```bash
# GitHub
OAUTH_GITHUB_CLIENT_ID=your_github_client_id
OAUTH_GITHUB_CLIENT_SECRET=your_github_client_secret

# Gmail (uses Google OAuth)
OAUTH_GMAIL_CLIENT_ID=your_google_client_id
OAUTH_GMAIL_CLIENT_SECRET=your_google_client_secret

# Google Drive (uses Google OAuth)
OAUTH_GOOGLE_DRIVE_CLIENT_ID=your_google_client_id
OAUTH_GOOGLE_DRIVE_CLIENT_SECRET=your_google_client_secret

# Slack
OAUTH_SLACK_CLIENT_ID=your_slack_client_id
OAUTH_SLACK_CLIENT_SECRET=your_slack_client_secret

# Notion
OAUTH_NOTION_CLIENT_ID=your_notion_client_id
OAUTH_NOTION_CLIENT_SECRET=your_notion_client_secret

# Dropbox
OAUTH_DROPBOX_CLIENT_ID=your_dropbox_client_id
OAUTH_DROPBOX_CLIENT_SECRET=your_dropbox_client_secret

# Discord
OAUTH_DISCORD_CLIENT_ID=your_discord_client_id
OAUTH_DISCORD_CLIENT_SECRET=your_discord_client_secret

# OAuth Redirect URI (where provider redirects after auth)
OAUTH_REDIRECT_URI=http://localhost:3000/oauth-callback
```

### Docker Compose Setup

Add environment variables to your `docker-compose.yml`:

```yaml
services:
  edison:
    environment:
      OAUTH_GITHUB_CLIENT_ID: ${OAUTH_GITHUB_CLIENT_ID}
      OAUTH_GITHUB_CLIENT_SECRET: ${OAUTH_GITHUB_CLIENT_SECRET}
      OAUTH_GMAIL_CLIENT_ID: ${OAUTH_GMAIL_CLIENT_ID}
      OAUTH_GMAIL_CLIENT_SECRET: ${OAUTH_GMAIL_CLIENT_SECRET}
      OAUTH_GOOGLE_DRIVE_CLIENT_ID: ${OAUTH_GOOGLE_DRIVE_CLIENT_ID}
      OAUTH_GOOGLE_DRIVE_CLIENT_SECRET: ${OAUTH_GOOGLE_DRIVE_CLIENT_SECRET}
      OAUTH_SLACK_CLIENT_ID: ${OAUTH_SLACK_CLIENT_ID}
      OAUTH_SLACK_CLIENT_SECRET: ${OAUTH_SLACK_CLIENT_SECRET}
      OAUTH_NOTION_CLIENT_ID: ${OAUTH_NOTION_CLIENT_ID}
      OAUTH_NOTION_CLIENT_SECRET: ${OAUTH_NOTION_CLIENT_SECRET}
      OAUTH_DROPBOX_CLIENT_ID: ${OAUTH_DROPBOX_CLIENT_ID}
      OAUTH_DROPBOX_CLIENT_SECRET: ${OAUTH_DROPBOX_CLIENT_SECRET}
      OAUTH_DISCORD_CLIENT_ID: ${OAUTH_DISCORD_CLIENT_ID}
      OAUTH_DISCORD_CLIENT_SECRET: ${OAUTH_DISCORD_CLIENT_SECRET}
      OAUTH_REDIRECT_URI: ${OAUTH_REDIRECT_URI:-http://localhost:3000/oauth-callback}
```

Then create a `.env` file with your actual credentials:

```bash
# .env file format
OAUTH_GITHUB_CLIENT_ID=github_123456
OAUTH_GITHUB_CLIENT_SECRET=github_secret_789
# ... etc
```

## Getting OAuth Credentials

### GitHub

1. Go to [GitHub Settings → Developer settings → OAuth Apps](https://github.com/settings/developers)
2. Click "New OAuth App"
3. Fill in:
   - **Application name**: EDISON
   - **Homepage URL**: `http://localhost:3000`
   - **Authorization callback URL**: `http://localhost:3000/oauth-callback`
4. Copy the **Client ID** and **Client Secret**
5. Set environment variables:
   ```bash
   OAUTH_GITHUB_CLIENT_ID=<Client ID>
   OAUTH_GITHUB_CLIENT_SECRET=<Client Secret>
   ```

### Gmail / Google Drive (Google OAuth)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable **Gmail API** and **Google Drive API**
4. Go to **Credentials** → **Create Credentials** → **OAuth 2.0 Client ID**
5. Choose **Web application**
6. Add authorized redirect URIs:
   - `http://localhost:3000/oauth-callback`
7. Copy **Client ID** and **Client Secret**
8. Set environment variables:
   ```bash
   OAUTH_GMAIL_CLIENT_ID=<Client ID>
   OAUTH_GMAIL_CLIENT_SECRET=<Client Secret>
   OAUTH_GOOGLE_DRIVE_CLIENT_ID=<Client ID>
   OAUTH_GOOGLE_DRIVE_CLIENT_SECRET=<Client Secret>
   ```

### Slack

1. Go to [Slack Apps](https://api.slack.com/apps)
2. Click **Create New App** → **From scratch**
3. Choose workspace
4. Go to **OAuth & Permissions** in left sidebar
5. Add **Redirect URLs**: `http://localhost:3000/oauth-callback`
6. Select scopes needed (chat:write, users:read, etc.)
7. Copy **Client ID** and **Client Secret** from top of page
8. Set environment variables:
   ```bash
   OAUTH_SLACK_CLIENT_ID=<Client ID>
   OAUTH_SLACK_CLIENT_SECRET=<Client Secret>
   ```

### Notion

1. Go to [Notion Integrations](https://www.notion.so/my-integrations)
2. Click **New Integration**
3. Fill in details and configure capabilities
4. Go to **Distribution** section
5. Add redirect URI: `http://localhost:3000/oauth-callback`
6. Copy **OAuth Client ID** and **OAuth Client Secret**
7. Set environment variables:
   ```bash
   OAUTH_NOTION_CLIENT_ID=<OAuth Client ID>
   OAUTH_NOTION_CLIENT_SECRET=<OAuth Client Secret>
   ```

### Dropbox

1. Go to [Dropbox App Console](https://www.dropbox.com/developers/apps)
2. Click **Create App**
3. Choose:
   - **API**: Scoped API
   - **Type**: Full Dropbox
4. Go to **OAuth 2** section
5. Add **Redirect URIs**: `http://localhost:3000/oauth-callback`
6. Copy **App key** and **App secret**
7. Set environment variables:
   ```bash
   OAUTH_DROPBOX_CLIENT_ID=<App key>
   OAUTH_DROPBOX_CLIENT_SECRET=<App secret>
   ```

### Discord

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click **New Application**
3. Go to **OAuth2** in sidebar
4. Add **Redirect**: `http://localhost:3000/oauth-callback`
5. Copy **Client ID** and **Client Secret**
6. Set environment variables:
   ```bash
   OAUTH_DISCORD_CLIENT_ID=<Client ID>
   OAUTH_DISCORD_CLIENT_SECRET=<Client Secret>
   ```

## API Endpoints

### Start OAuth Flow

**Endpoint:** `POST /integrations/connectors/oauth-start/{provider}`

**Parameters:**
- `provider` (path): Provider name (github, gmail, google-drive, slack, notion, dropbox, discord)
- `redirect_uri` (optional, body): Custom redirect URI (defaults to OAUTH_REDIRECT_URI env var)

**Response:**
```json
{
  "ok": true,
  "provider": "github",
  "auth_url": "https://github.com/login/oauth/authorize?client_id=...",
  "state": "uuid-string"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/integrations/connectors/oauth-start/github
```

### Handle OAuth Callback

**Endpoint:** `POST /integrations/connectors/oauth-callback`

**Parameters (from OAuth provider):**
- `code` (query): Authorization code from provider
- `state` (query): State token to prevent CSRF
- `connector_name` (body): Name to save connector as

**Response:**
```json
{
  "ok": true,
  "connector": "my-github",
  "provider": "github",
  "message": "OAuth authorization successful"
}
```

### Get Provider Auth Details

**Endpoint:** `GET /integrations/connectors/auth/{provider}`

**Response for OAuth2 provider:**
```json
{
  "ok": true,
  "provider": "github",
  "label": "GitHub",
  "auth_type": "oauth2",
  "oauth_auth_url": "https://github.com/login/oauth/authorize",
  "oauth_token_url": "https://github.com/login/oauth/access_token",
  "oauth_scopes": ["repo", "user", "gist"],
  "setup_steps": [
    "1. Click 'Connect with GitHub' button",
    "2. You'll be redirected to authenticate",
    "3. Authorize EDISON to access your account",
    "4. You'll be redirected back and token will be saved"
  ]
}
```

## Token Storage

Access tokens are stored securely in `config/integrations/connectors.json` with metadata:

```json
{
  "connectors": [
    {
      "name": "my-github",
      "provider": "github",
      "enabled": true,
      "auth_type": "bearer",
      "token": "gho_16C7e42F292c6912E7...",
      "token_type": "Bearer",
      "expires_at": 1234567890,
      "refresh_token": "ghr_1B4a2e77838347a7E...",
      "base_url": "https://api.github.com"
    }
  ]
}
```

### Token Refresh

Some providers (Google, Slack, Notion) support refresh tokens. If a token expires, EDISON will automatically:
1. Check `expires_at` timestamp
2. Use `refresh_token` to get new access token
3. Update stored token with new expiry

## Security Considerations

### CSRF Protection
- All OAuth flows use state tokens
- State tokens are validated before exchanging code for token
- State tokens expire after 10 minutes
- Mismatch between request state and response state rejects request

### Token Security
- Tokens are stored locally in `config/integrations/connectors.json`
- File permissions restrict access to owner only
- Tokens are never logged or exposed in responses
- When retrieving connector details, token values are omitted

### Best Practices
1. **Use HTTPS in production**: Set `OAUTH_REDIRECT_URI` to https URL
2. **Keep secrets secure**: Don't commit `.env` files to Git
3. **Rotate credentials periodically**: Regenerate Client ID/Secret in provider settings
4. **Monitor token usage**: Check provider's activity log for suspicious access
5. **Use minimal scopes**: Request only scopes your integration needs

## Troubleshooting

### "OAUTH_GITHUB_CLIENT_ID not configured"
**Cause:** Environment variable not set
**Solution:**
```bash
export OAUTH_GITHUB_CLIENT_ID=your_client_id
export OAUTH_GITHUB_CLIENT_SECRET=your_secret
# Or add to docker-compose.yml environment
```

### "Invalid or expired state token"
**Cause:** Too much time between oauth-start and oauth-callback (>10 min), or CSRF attack attempt
**Solution:** 
1. Ensure system clocks are synchronized
2. Try again (state tokens are single-use)
3. Check browser logs for errors during redirect

### "No access token in response"
**Cause:** Provider returned error, invalid code, or scope mismatch
**Solution:**
1. Check provider's logs for what went wrong
2. Verify redirect URI matches exactly in provider settings
3. Verify requested scopes are valid for app configuration

### Provider returns "Invalid client_id"
**Cause:** Credentials don't match registered app, or app was deleted
**Solution:**
1. Verify Client ID and Client Secret in environment variables
2. Regenerate credentials in provider's dashboard if app was modified
3. Check provider's console to ensure app is still active

## Frontend Integration

### HTML Button Example
```html
<button class="connect-btn" onclick="startOAuth('github')">
  Connect with GitHub
</button>

<script>
async function startOAuth(provider) {
  try {
    // Get the auth URL
    const response = await fetch(`/integrations/connectors/oauth-start/${provider}`, {
      method: 'POST'
    });
    const data = await response.json();
    
    if (data.ok) {
      // Redirect to provider's OAuth page
      window.location.href = data.auth_url;
    } else {
      alert('Error: ' + data.detail);
    }
  } catch (error) {
    alert('Connection failed: ' + error.message);
  }
}
</script>
```

### Handling Callback
```javascript
// On page that handles OAuth redirect (default: http://localhost:3000/oauth-callback)
async function handleOAuthCallback() {
  const params = new URLSearchParams(window.location.search);
  const code = params.get('code');
  const state = params.get('state');
  
  try {
    const response = await fetch('/integrations/connectors/oauth-callback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        code: code,
        state: state,
        connector_name: 'my-' + params.get('provider') // or get from session
      })
    });
    
    const result = await response.json();
    if (result.ok) {
      alert(`Successfully connected ${result.provider}!`);
      // Redirect to connectors page
      window.location.href = '/connectors';
    } else {
      alert('Authorization failed: ' + result.detail);
    }
  } catch (error) {
    alert('Callback failed: ' + error.message);
  }
}

// Call when page loads
window.addEventListener('load', handleOAuthCallback);
```

## Testing OAuth Flow

### Using curl (linux/mac)
```bash
# 1. Start OAuth for GitHub
curl -X POST http://localhost:8000/integrations/connectors/oauth-start/github

# Response will include auth_url and state token
# 2. Manually visit the auth_url in browser, authorize
# 3. Get redirected back with code and state
# 4. Exchange code for token (usually handled by frontend)
curl -X POST http://localhost:8000/integrations/connectors/oauth-callback \
  -H "Content-Type: application/json" \
  -d '{
    "code": "authorization_code_from_provider",
    "state": "state_token_from_step_1",
    "connector_name": "my-github"
  }'
```

## Next Steps

1. Set up OAuth credentials for your providers (see "Getting OAuth Credentials" section)
2. Configure environment variables
3. Update frontend UI to show OAuth buttons for supported providers
4. Test the OAuth flow end-to-end
5. Monitor token expiration and refresh

For specific provider documentation, see their OAuth documentation links at [Supported Providers](#supported-providers).
