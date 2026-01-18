# Deployment Guide

This guide will help you deploy the Predictive Scaling Dashboard to various platforms.

## Prerequisites

1. Backend API deployed and accessible (Railway, Render, AWS, etc.)
2. Backend API URL (e.g., `https://your-api.railway.app`)

## Deploy to Vercel (Recommended)

Vercel is the easiest way to deploy Next.js applications.

### Option 1: Deploy via GitHub

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add dashboard"
   git push origin main
   ```

2. **Import to Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Click "Add New Project"
   - Import your GitHub repository
   - Select the `dashboard` directory as the root directory

3. **Configure Environment Variables**
   - Add `NEXT_PUBLIC_API_URL` with your backend URL
   - Example: `https://predictive-scaling-api.railway.app`

4. **Deploy**
   - Click "Deploy"
   - Your dashboard will be live at `https://your-project.vercel.app`

### Option 2: Deploy via CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Navigate to dashboard directory
cd dashboard

# Deploy
vercel

# Follow the prompts and set environment variables when asked
```

## Deploy to Netlify

1. **Build Settings**
   - Build command: `npm run build`
   - Publish directory: `.next`
   - Base directory: `dashboard`

2. **Environment Variables**
   ```
   NEXT_PUBLIC_API_URL=https://your-backend-api.com
   ```

3. **Deploy**
   ```bash
   # Install Netlify CLI
   npm i -g netlify-cli

   # Build and deploy
   cd dashboard
   npm run build
   netlify deploy --prod
   ```

## Deploy to Railway

1. **Create `railway.toml`** in the dashboard directory:
   ```toml
   [build]
   builder = "nixpacks"
   buildCommand = "npm install && npm run build"

   [deploy]
   startCommand = "npm start"
   restartPolicyType = "on_failure"
   restartPolicyMaxRetries = 10

   [[services]]
   name = "dashboard"

   [services.domains]
   domains = ["your-domain.railway.app"]
   ```

2. **Deploy**
   ```bash
   # Install Railway CLI
   npm i -g @railway/cli

   # Login
   railway login

   # Initialize and deploy
   cd dashboard
   railway init
   railway up
   ```

3. **Set Environment Variables**
   - Go to Railway dashboard
   - Add `NEXT_PUBLIC_API_URL` variable

## Deploy with Docker

1. **Create `Dockerfile`** in dashboard directory:
   ```dockerfile
   FROM node:18-alpine AS base

   # Install dependencies only when needed
   FROM base AS deps
   WORKDIR /app
   COPY package*.json ./
   RUN npm ci

   # Rebuild the source code only when needed
   FROM base AS builder
   WORKDIR /app
   COPY --from=deps /app/node_modules ./node_modules
   COPY . .
   ENV NEXT_TELEMETRY_DISABLED 1
   RUN npm run build

   # Production image, copy all files and run next
   FROM base AS runner
   WORKDIR /app
   ENV NODE_ENV production
   ENV NEXT_TELEMETRY_DISABLED 1

   RUN addgroup --system --gid 1001 nodejs
   RUN adduser --system --uid 1001 nextjs

   COPY --from=builder /app/public ./public
   COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
   COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

   USER nextjs
   EXPOSE 3000
   ENV PORT 3000

   CMD ["node", "server.js"]
   ```

2. **Update `next.config.mjs`**:
   ```javascript
   const nextConfig = {
     output: 'standalone',
     // ... rest of config
   };
   ```

3. **Build and run**:
   ```bash
   docker build -t predictive-scaling-dashboard .
   docker run -p 3000:3000 -e NEXT_PUBLIC_API_URL=https://your-api.com predictive-scaling-dashboard
   ```

## Deploy to AWS Amplify

1. **Connect Repository**
   - Go to AWS Amplify Console
   - Connect your GitHub repository
   - Select the `dashboard` directory as the root

2. **Build Settings** (amplify.yml):
   ```yaml
   version: 1
   applications:
     - frontend:
         phases:
           preBuild:
             commands:
               - cd dashboard
               - npm ci
           build:
             commands:
               - npm run build
         artifacts:
           baseDirectory: dashboard/.next
           files:
             - '**/*'
         cache:
           paths:
             - dashboard/node_modules/**/*
   ```

3. **Environment Variables**
   - Add `NEXT_PUBLIC_API_URL` in Amplify Console

## Deploy to Azure Static Web Apps

1. **GitHub Actions Workflow** (.github/workflows/azure-static-web-apps.yml):
   ```yaml
   name: Azure Static Web Apps CI/CD

   on:
     push:
       branches:
         - main

   jobs:
     build_and_deploy_job:
       runs-on: ubuntu-latest
       name: Build and Deploy Job
       steps:
         - uses: actions/checkout@v3

         - name: Build And Deploy
           uses: Azure/static-web-apps-deploy@v1
           with:
             azure_static_web_apps_api_token: ${{ secrets.AZURE_STATIC_WEB_APPS_API_TOKEN }}
             repo_token: ${{ secrets.GITHUB_TOKEN }}
             action: "upload"
             app_location: "/dashboard"
             api_location: ""
             output_location: ".next"
             app_build_command: "npm run build"
   ```

2. **Set Environment Variables** in Azure Portal

## Custom Domain Setup

### Vercel
1. Go to Project Settings → Domains
2. Add your custom domain
3. Update DNS records as instructed

### Netlify
1. Go to Site Settings → Domain Management
2. Add custom domain
3. Update DNS records

### Railway
1. Go to Service → Settings → Domains
2. Add custom domain
3. Update DNS CNAME record

## SSL/HTTPS

All recommended platforms (Vercel, Netlify, Railway) provide automatic SSL certificates via Let's Encrypt. No additional configuration needed!

## Performance Optimization

### Enable Compression
All platforms enable gzip/brotli compression by default.

### CDN
- Vercel: Global Edge Network (automatic)
- Netlify: Global CDN (automatic)
- Cloudflare: Can be added in front of any deployment

### Caching Headers
Headers are automatically optimized by Next.js:
- Static assets: 1 year cache
- Pages: Revalidated as needed
- API calls: No cache (controlled by hooks)

## Monitoring

### Vercel Analytics
Add to your dashboard:
```bash
npm install @vercel/analytics
```

In `app/layout.tsx`:
```tsx
import { Analytics } from '@vercel/analytics/react';

export default function RootLayout({ children }) {
  return (
    <html>
      <body>
        {children}
        <Analytics />
      </body>
    </html>
  );
}
```

### Error Tracking
Consider adding Sentry:
```bash
npm install @sentry/nextjs
```

## Troubleshooting

### Dashboard shows "Failed to load" errors
- Ensure `NEXT_PUBLIC_API_URL` is set correctly
- Check if backend API is accessible from the internet
- Verify CORS is enabled on the backend for your dashboard domain

### Build fails
- Check Node.js version (should be 18+)
- Clear `.next` and `node_modules`, reinstall dependencies
- Check for TypeScript errors: `npm run lint`

### Environment variables not working
- Remember: `NEXT_PUBLIC_*` variables are embedded at build time
- Redeploy after changing environment variables
- Client-side variables must start with `NEXT_PUBLIC_`

## Post-Deployment Checklist

- [ ] Dashboard loads successfully
- [ ] API connection works
- [ ] All animations are smooth
- [ ] Responsive design works on mobile
- [ ] Custom domain configured (if applicable)
- [ ] SSL certificate is active
- [ ] Footer links updated with your information
- [ ] README.md updated with live URL
- [ ] Share with recruiters!

## Support

For issues specific to:
- Next.js: [nextjs.org/docs](https://nextjs.org/docs)
- Vercel: [vercel.com/docs](https://vercel.com/docs)
- Railway: [docs.railway.app](https://docs.railway.app)

---

Need help? Open an issue on GitHub or reach out via the contact info in the footer.
