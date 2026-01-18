# Predictive Scaling Dashboard

A stunning, modern dashboard for visualizing and interacting with an ML-powered auto-scaling system. Built with Next.js, TypeScript, Tailwind CSS, and Framer Motion.

![Dashboard Preview](./preview.png)

## Features

- **Live Status Panel** - Real-time monitoring of system health, replicas, CPU, and memory usage
- **Predictive Analytics Chart** - Time-series visualization with confidence intervals (P10/P50/P90)
- **Interactive Demo** - Trigger ML predictions and simulate traffic spikes
- **Scaling Timeline** - Visual history of all scaling decisions and actions
- **Architecture Diagram** - Animated system architecture flow
- **Responsive Design** - Works perfectly on desktop, tablet, and mobile
- **Dark Theme** - Modern, glassmorphism aesthetic inspired by Linear and Vercel

## Tech Stack

- **Framework**: Next.js 14 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Animations**: Framer Motion
- **Charts**: Recharts
- **Icons**: Lucide React
- **API Client**: Axios
- **Date Handling**: date-fns

## Prerequisites

- Node.js 18+ installed
- Backend API running at `http://localhost:8000` (or configure `NEXT_PUBLIC_API_URL`)

## Getting Started

### 1. Install Dependencies

```bash
npm install
```

### 2. Configure Environment Variables

Create a `.env.local` file:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

For production deployment (e.g., Vercel), set this to your Railway/Render backend URL:

```env
NEXT_PUBLIC_API_URL=https://your-api.railway.app
```

### 3. Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the dashboard.

## Project Structure

```
dashboard/
├── app/
│   ├── layout.tsx          # Root layout with fonts and metadata
│   ├── page.tsx            # Main dashboard page
│   └── globals.css         # Global styles and animations
├── components/
│   ├── HeroSection.tsx     # Hero with features
│   ├── LiveStatusPanel.tsx # Real-time status monitoring
│   ├── PredictionsChart.tsx # Time-series predictions
│   ├── InteractiveDemo.tsx # Demo controls
│   ├── ScalingTimeline.tsx # Scaling history
│   ├── ArchitectureDiagram.tsx # System architecture
│   └── Footer.tsx          # Footer with links
├── hooks/
│   └── useApi.ts           # Custom hooks for API calls
├── lib/
│   └── api.ts              # API client and types
└── public/                 # Static assets
```

## API Integration

The dashboard connects to the backend API and expects the following endpoints:

- `GET /health` - Health check
- `GET /api/v1/scaling/status` - Current scaling status
- `GET /api/v1/predictions/current` - ML predictions
- `POST /api/v1/predictions/trigger` - Trigger new prediction
- `GET /api/v1/scaling/decisions` - Scaling history
- `GET /api/v1/events` - Business events
- `POST /api/v1/events` - Create event
- `GET /api/v1/config` - System configuration

## Deployment

### Deploy to Vercel (Recommended)

1. Push your code to GitHub
2. Import the project to Vercel
3. Set environment variable: `NEXT_PUBLIC_API_URL=https://your-backend-api.com`
4. Deploy!

```bash
npm run build
npm run start
```

### Deploy to Other Platforms

The dashboard is a standard Next.js app and can be deployed to:
- Netlify
- Railway
- AWS Amplify
- Azure Static Web Apps
- Self-hosted with Docker

## Customization

### Update Personal Info

Edit `components/Footer.tsx` to update:
- Your name
- GitHub link
- LinkedIn link
- Email

### Change Color Scheme

Edit `app/globals.css` to modify the color variables:

```css
:root {
  --background: 222.2 84% 4.9%;
  --foreground: 210 40% 98%;
  --primary: 210 40% 98%;
  /* ... */
}
```

### Modify Animations

All animations use Framer Motion. Adjust timing and easing in component files:

```tsx
<motion.div
  initial={{ opacity: 0, y: 20 }}
  animate={{ opacity: 1, y: 0 }}
  transition={{ duration: 0.5 }}
>
```

## Performance Optimization

- API calls are cached and refreshed at optimal intervals
- Animations use GPU-accelerated properties (transform, opacity)
- Charts render efficiently with Recharts
- Components are lazy-loaded where possible

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers (iOS Safari, Chrome Mobile)

## License

MIT

## Author

Built with care for the recruiters. This is a portfolio project showcasing modern web development and system design skills.

---

For more information about the backend system, see the main [README.md](../README.md) in the project root.
