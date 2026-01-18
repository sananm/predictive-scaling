# Quick Start Guide

Get the dashboard running in under 5 minutes!

## Prerequisites

- Node.js 18+ installed ([download here](https://nodejs.org/))
- Backend API accessible (running locally or deployed)

## Installation

### Option 1: Automated Setup (Recommended)

```bash
cd dashboard
./setup.sh
```

This script will:
- Check Node.js version
- Install all dependencies
- Create .env.local with defaults
- Guide you through next steps

### Option 2: Manual Setup

```bash
cd dashboard

# Install dependencies
npm install

# Create environment file
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# Start development server
npm run dev
```

## Running the Dashboard

### 1. Start Your Backend API

Make sure your backend API is running:

```bash
# In the project root directory
make run-api
# Or
python -m uvicorn src.api.main:app --reload
```

Verify it's working: `curl http://localhost:8000/health`

### 2. Start the Dashboard

```bash
cd dashboard
npm run dev
```

### 3. Open in Browser

Visit [http://localhost:3000](http://localhost:3000)

You should see the dashboard with:
- Live status panel showing current replicas, CPU, and memory
- Predictions chart (may be empty until you trigger a prediction)
- Interactive demo buttons
- Scaling timeline
- Architecture diagram

## First Steps

### 1. Check Connection
Look at the "Live Status Panel" - if you see data, the API connection is working!

### 2. Trigger a Prediction
Click the "Trigger Prediction" button in the Interactive Demo section. This will:
- Run the ML models
- Generate predictions for the next hour
- Update the predictions chart

### 3. Simulate Traffic Spike
Click "Simulate Traffic Spike" to:
- Create a flash sale event
- Auto-trigger predictions
- Watch the system respond

### 4. Explore the Timeline
Scroll down to see the scaling decisions timeline, showing when and why the system scaled.

## Troubleshooting

### "Failed to load status" errors

**Problem**: Dashboard can't connect to the API

**Solutions**:
1. Check if backend API is running: `curl http://localhost:8000/health`
2. Verify `.env.local` has the correct URL
3. Check for CORS issues in browser console
4. Ensure no firewall is blocking port 8000

### Dashboard loads but shows no data

**Problem**: API is running but returns empty responses

**Solutions**:
1. Trigger a prediction manually: `curl -X POST http://localhost:8000/api/v1/predictions/trigger`
2. Check if ML models are trained: `ls models/`
3. Run training script: `python scripts/train_models.py`
4. Check API logs for errors

### npm install fails

**Problem**: Dependency installation errors

**Solutions**:
1. Delete `node_modules` and `package-lock.json`
2. Run `npm cache clean --force`
3. Run `npm install` again
4. Check Node.js version: `node -v` (must be 18+)

### Port 3000 already in use

**Problem**: Another process is using port 3000

**Solutions**:
```bash
# Kill process on port 3000 (macOS/Linux)
lsof -ti:3000 | xargs kill -9

# Or run on different port
PORT=3001 npm run dev
```

### Animations are laggy

**Problem**: Performance issues

**Solutions**:
1. Close other browser tabs
2. Disable browser extensions temporarily
3. Check system resources
4. Try Chrome/Edge (best performance with GPU acceleration)

## Development Tips

### Hot Reload
The dashboard uses Next.js hot reload - any changes you make to files will automatically refresh in the browser.

### Component Structure
- `components/` - All UI components
- `hooks/` - Custom React hooks for API calls
- `lib/` - Utilities and API client
- `app/` - Next.js pages and layouts

### Making Changes

**Change API URL**:
Edit `.env.local`:
```env
NEXT_PUBLIC_API_URL=https://your-deployed-api.com
```

**Update Personal Info**:
Edit `components/Footer.tsx` to add your GitHub, LinkedIn, and email.

**Customize Colors**:
Edit `app/globals.css` to change the color scheme.

**Add New Components**:
Create in `components/` directory and import in `app/page.tsx`.

## Build for Production

```bash
npm run build
npm start
```

This creates an optimized production build and starts the server on port 3000.

## Deploy

See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed deployment instructions for:
- Vercel (recommended)
- Netlify
- Railway
- Docker
- AWS Amplify
- Azure

Quick deploy to Vercel:
```bash
npm i -g vercel
vercel
```

## Next Steps

1. **Customize the dashboard** - Update colors, add your info, tweak animations
2. **Deploy to production** - Use Vercel for easiest deployment
3. **Share with recruiters** - Add the live URL to your resume!
4. **Explore the code** - See how React, TypeScript, and Framer Motion work together

## Getting Help

- **API Issues**: Check the main project README.md
- **Dashboard Issues**: Open an issue on GitHub
- **Deployment**: See DEPLOYMENT.md
- **Customization**: See README.md

## Resources

- [Next.js Docs](https://nextjs.org/docs)
- [Framer Motion Docs](https://www.framer.com/motion/)
- [Tailwind CSS Docs](https://tailwindcss.com/docs)
- [Recharts Docs](https://recharts.org/)

---

Happy coding! Make this dashboard yours and impress those recruiters! 🚀
