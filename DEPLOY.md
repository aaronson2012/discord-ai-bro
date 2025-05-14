# Deploying Discord AI Bro to Fly.io

This guide will walk you through deploying your Discord AI Bro bot to Fly.io.

## Prerequisites

1. A Fly.io account (sign up at https://fly.io)
2. Fly CLI installed on your machine
3. Your Discord bot token, Google API key, and Supabase credentials

## Installation Steps

### 1. Install Fly CLI

If you haven't already installed the Fly CLI, follow these instructions:

**On macOS:**
```bash
brew install flyctl
```

**On Linux:**
```bash
curl -L https://fly.io/install.sh | sh
```

**On Windows:**
```
powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
```

### 2. Login to Fly.io

```bash
fly auth login
```

### 3. Deploy the Application

From the root directory of your project:

```bash
# Launch the app (first-time deployment)
fly launch
```

This will:
- Detect your Dockerfile
- Ask you to name your app (or use the default "discord-ai-bro")
- Ask which region to deploy to
- Ask if you want to set up a PostgreSQL database (not needed if you're using Supabase)
- Ask if you want to deploy now

### 4. Set Environment Variables

Set all the required environment variables:

```bash
fly secrets set DISCORD_BOT_TOKEN=your_discord_bot_token
fly secrets set GOOGLE_API_KEY=your_google_api_key
fly secrets set GEMINI_MODEL_NAME="gemini-1.5-flash-latest"
fly secrets set SUPABASE_URL=your_supabase_url
fly secrets set SUPABASE_KEY=your_supabase_anon_key
fly secrets set API_TIMEOUT_SECONDS=30
fly secrets set GEMINI_TIMEOUT_SECONDS=45
fly secrets set SUPABASE_TIMEOUT_SECONDS=20
```

### 5. Deploy or Redeploy

If you chose not to deploy in step 3, or if you need to redeploy after changes:

```bash
fly deploy
```

### 6. Check Deployment Status

```bash
# Check app status
fly status

# View logs
fly logs
```

### 7. Scale Your App (Optional)

By default, Fly.io will deploy your app with minimal resources. You can scale up if needed:

```bash
# Scale to a larger VM
fly scale vm shared-cpu-1x 1GB
```

## Monitoring and Management

### View Logs

```bash
fly logs
```

### SSH into the VM

```bash
fly ssh console
```

### Check Health Status

The bot has a health check endpoint at `/health`. You can access it via:

```
https://discord-ai-bro.fly.dev/health
```

### Restart the App

```bash
fly apps restart discord-ai-bro
```

## Troubleshooting

### Common Issues

1. **Bot not connecting to Discord:**
   - Check logs with `fly logs`
   - Verify your DISCORD_BOT_TOKEN is set correctly

2. **Health check failing:**
   - The health check endpoint should return a 200 status code
   - If it's failing, check the logs for errors

3. **Memory issues:**
   - If you see out-of-memory errors, consider scaling up your VM

### Getting Help

If you encounter issues, you can:
- Check Fly.io documentation: https://fly.io/docs/
- Visit Fly.io community forum: https://community.fly.io/
- Check Discord.py documentation: https://discordpy.readthedocs.io/

## Updating Your Bot

When you make changes to your bot:

1. Commit your changes to your repository
2. Run `fly deploy` to deploy the updated version

Fly.io will build a new image and deploy it automatically.
