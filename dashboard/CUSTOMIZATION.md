# Customization Guide

Make this dashboard your own! This guide shows you how to customize every aspect.

## Quick Customizations (5 minutes)

### 1. Update Personal Information

**File**: `components/Footer.tsx`

```tsx
const links = [
  { icon: Github, label: 'GitHub', href: 'https://github.com/YOUR_USERNAME' },
  { icon: Linkedin, label: 'LinkedIn', href: 'https://linkedin.com/in/YOUR_USERNAME' },
  { icon: Mail, label: 'Email', href: 'mailto:YOUR_EMAIL@example.com' },
];

// And update the name:
<span className="text-purple-400 font-semibold">Your Name</span>
```

### 2. Change API URL

**File**: `.env.local`

```env
# For local development
NEXT_PUBLIC_API_URL=http://localhost:8000

# For production
NEXT_PUBLIC_API_URL=https://your-api.railway.app
```

### 3. Update Page Title

**File**: `app/layout.tsx`

```tsx
export const metadata: Metadata = {
  title: "Your Project Name - Predictive Scaling",
  description: "Your custom description here",
};
```

---

## Color Scheme (10 minutes)

### Change Primary Colors

**File**: `app/globals.css`

```css
:root {
  /* Background - the main dark color */
  --background: 222.2 84% 4.9%;  /* Very dark blue */

  /* Foreground - text color */
  --foreground: 210 40% 98%;     /* Almost white */

  /* Primary - main accent color */
  --primary: 210 40% 98%;

  /* Secondary - secondary elements */
  --secondary: 217.2 32.6% 17.5%; /* Dark gray-blue */
}
```

### Popular Color Schemes

**Purple Theme** (Current):
```css
/* Gradients in components use purple/pink/blue */
from-purple-500 to-pink-500
from-purple-400 via-pink-400 to-blue-400
```

**Cyberpunk/Neon**:
```css
/* Replace purple with cyan/magenta */
from-cyan-500 to-magenta-500
from-cyan-400 via-purple-400 to-magenta-400
```

**Corporate Blue**:
```css
/* Use blue/teal gradients */
from-blue-500 to-teal-500
from-blue-400 via-cyan-400 to-teal-400
```

**Dark Green/Matrix**:
```css
/* Use green/emerald */
from-green-500 to-emerald-500
from-green-400 via-emerald-400 to-teal-400
```

### Apply Color Scheme Globally

Find and replace in all component files:
- `purple` → your color
- `pink` → your accent
- `blue` → your secondary

---

## Animation Customization

### Change Animation Speed

**In any component file**:

```tsx
// Slower animations (more elegant)
transition={{ duration: 0.8 }}  // Instead of 0.5

// Faster animations (more snappy)
transition={{ duration: 0.3 }}  // Instead of 0.5

// Different easing
transition={{ ease: "easeOut" }}
transition={{ ease: [0.6, 0.05, 0.01, 0.9] }} // Custom cubic bezier
```

### Disable Specific Animations

```tsx
// Change from:
<motion.div
  initial={{ opacity: 0, y: 20 }}
  animate={{ opacity: 1, y: 0 }}
>

// To:
<motion.div
  initial={{ opacity: 1, y: 0 }}
  animate={{ opacity: 1, y: 0 }}
>
```

### Change Hover Effects

```tsx
// More subtle
whileHover={{ scale: 1.02 }}

// More dramatic
whileHover={{ scale: 1.1, rotate: 5 }}

// Add glow
whileHover={{
  scale: 1.05,
  boxShadow: "0 0 30px rgba(139, 92, 246, 0.6)"
}}
```

---

## Layout Customization

### Change Component Order

**File**: `app/page.tsx`

```tsx
// Reorder sections by moving components around:
<main>
  <HeroSection />
  <LiveStatusPanel />       // Swap these
  <PredictionsChart />      // to change order
  <InteractiveDemo />
  <ScalingTimeline />
  <ArchitectureDiagram />
</main>
```

### Change Grid Layouts

```tsx
// Two columns instead of three
<div className="grid grid-cols-1 md:grid-cols-2 gap-8">

// Four columns on desktop
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">

// Single column (no grid)
<div className="space-y-8">
```

### Adjust Spacing

```tsx
// More space between sections
<div className="space-y-12">  // Instead of space-y-8

// Wider container
<div className="max-w-7xl">   // Instead of max-w-6xl

// More padding
<div className="p-12">        // Instead of p-8
```

---

## Component Customization

### Add New Feature Card to Hero

**File**: `components/HeroSection.tsx`

```tsx
const features = [
  // ... existing features ...
  {
    icon: YourIcon,  // Import from lucide-react
    title: 'Your Feature',
    description: 'Your description',
  },
];
```

### Customize Status Panel Metrics

**File**: `components/LiveStatusPanel.tsx`

```tsx
// Change gauge colors
const cpuColor = data && data.cpu_usage > 90 ? '#ef4444' : '#8b5cf6';

// Add new metric
<div className="text-center">
  <div className="text-4xl font-bold">
    {data?.requests_per_second || 0}
  </div>
  <p className="text-sm text-gray-400">Requests/sec</p>
</div>
```

### Modify Chart Appearance

**File**: `components/PredictionsChart.tsx`

```tsx
// Change line thickness
strokeWidth={5}  // Instead of 3

// Change line type
type="monotone"  // or "linear", "step", "natural"

// Add more data series
<Line
  dataKey="your_metric"
  stroke="#your_color"
  strokeWidth={2}
/>
```

---

## Text and Content

### Update Hero Headline

**File**: `components/HeroSection.tsx`

```tsx
<h1 className="text-6xl md:text-7xl font-bold mb-6">
  Your Custom Headline
</h1>

<p className="text-xl md:text-2xl text-gray-300">
  Your custom description and value proposition
</p>
```

### Change Tech Stack Pills

```tsx
{['Your', 'Custom', 'Tech', 'Stack'].map((tech) => (
  <span key={tech}>{tech}</span>
))}
```

### Update Architecture Steps

**File**: `components/ArchitectureDiagram.tsx`

```tsx
<ComponentCard
  icon={YourIcon}
  title="Your Step"
  subtitle="Your description"
  gradient="from-color-500/20 to-color-500/20"
/>
```

---

## Advanced Customizations

### Add New API Endpoint

**File**: `lib/api.ts`

```tsx
// 1. Add type
export interface YourDataType {
  field1: string;
  field2: number;
}

// 2. Add function
export const apiClient = {
  // ... existing functions ...

  getYourData: async (): Promise<YourDataType> => {
    const response = await api.get('/api/v1/your-endpoint');
    return response.data;
  },
};
```

**File**: `hooks/useApi.ts`

```tsx
// 3. Add hook
export const useYourData = (interval: number = 5000) => {
  const [data, setData] = useState<YourDataType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const result = await apiClient.getYourData();
        setData(result);
        setError(null);
      } catch (err) {
        setError('Failed to fetch data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const intervalId = setInterval(fetchData, interval);
    return () => clearInterval(intervalId);
  }, [interval]);

  return { data, loading, error };
};
```

**File**: `components/YourComponent.tsx`

```tsx
// 4. Use in component
import { useYourData } from '@/hooks/useApi';

export default function YourComponent() {
  const { data, loading, error } = useYourData();

  return (
    <div>
      {data && <p>{data.field1}</p>}
    </div>
  );
}
```

### Add New Chart Type

```tsx
import { BarChart, Bar, PieChart, Pie } from 'recharts';

// Bar chart
<BarChart data={chartData}>
  <Bar dataKey="value" fill="#8b5cf6" />
</BarChart>

// Pie chart
<PieChart>
  <Pie data={chartData} dataKey="value" fill="#8b5cf6" />
</PieChart>
```

### Create Custom Animation

```tsx
import { motion } from 'framer-motion';

const customVariants = {
  hidden: { opacity: 0, scale: 0.8 },
  visible: {
    opacity: 1,
    scale: 1,
    transition: {
      type: "spring",
      stiffness: 100,
      damping: 10
    }
  }
};

<motion.div
  variants={customVariants}
  initial="hidden"
  animate="visible"
>
  Your content
</motion.div>
```

---

## Styling Customizations

### Change Font

**File**: `app/layout.tsx`

```tsx
import { Inter, Poppins, Roboto } from "next/font/google";

const poppins = Poppins({
  subsets: ["latin"],
  weight: ['400', '600', '700']
});

// Use in body
<body className={poppins.className}>
```

### Add Shadows

```tsx
// Subtle shadow
className="shadow-lg"

// Colored shadow
className="shadow-lg shadow-purple-500/50"

// Large glow
className="shadow-2xl shadow-purple-500/30"
```

### Border Radius

```tsx
// More rounded
className="rounded-3xl"  // Instead of rounded-2xl

// Fully rounded
className="rounded-full"

// Sharp corners
className="rounded-none"
```

### Background Effects

```tsx
// Add blur
className="backdrop-blur-xl"

// Adjust opacity
className="bg-white/10"  // 10% opacity

// Add gradient overlay
<div className="bg-gradient-to-br from-purple-900/50 to-blue-900/50" />
```

---

## Responsive Breakpoints

### Customize for Your Needs

```tsx
// Mobile first (default)
className="text-sm md:text-base lg:text-lg"

// Tablet specific
className="md:block lg:hidden"

// Custom breakpoint (in tailwind.config.ts)
theme: {
  screens: {
    'xs': '475px',
    'sm': '640px',
    'md': '768px',
    'lg': '1024px',
    'xl': '1280px',
    '2xl': '1536px',
    '3xl': '1920px',
  },
}
```

---

## Performance Tuning

### Adjust Refresh Rates

```tsx
// Slower updates (less API calls)
const { data } = useScalingStatus(10000);  // 10 seconds

// Faster updates (more real-time)
const { data } = useScalingStatus(1000);   // 1 second
```

### Lazy Load Components

```tsx
import dynamic from 'next/dynamic';

const HeavyComponent = dynamic(() => import('./HeavyComponent'), {
  loading: () => <p>Loading...</p>,
});
```

### Reduce Motion for Performance

```tsx
// In component
const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

<motion.div
  animate={prefersReducedMotion ? {} : { scale: 1.05 }}
>
```

---

## Adding New Pages

**File**: `app/about/page.tsx`

```tsx
export default function AboutPage() {
  return (
    <main className="min-h-screen animated-gradient p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold mb-6">About</h1>
        <p>Your content here</p>
      </div>
    </main>
  );
}
```

**Navigation** (create `components/Nav.tsx`):

```tsx
import Link from 'next/link';

export default function Nav() {
  return (
    <nav className="fixed top-0 left-0 right-0 glass p-4 z-50">
      <div className="max-w-7xl mx-auto flex justify-between">
        <Link href="/">Dashboard</Link>
        <Link href="/about">About</Link>
      </div>
    </nav>
  );
}
```

---

## Testing Customizations

### Preview Changes

```bash
npm run dev
```

Open http://localhost:3000 and see your changes in real-time.

### Test Responsive Design

- Chrome DevTools → Device Toolbar (Cmd+Shift+M)
- Test on: iPhone, iPad, Desktop
- Check all breakpoints

### Test Animations

- Watch for smooth 60fps
- Check in different browsers
- Test with slow motion (Chrome DevTools → Animations)

### Test API Integration

- Start backend API
- Trigger different states
- Verify error handling

---

## Common Customization Patterns

### Change Card Style

From glassmorphism to solid:
```tsx
// From:
className="glass rounded-2xl"

// To:
className="bg-gray-900 border border-gray-800 rounded-2xl"
```

### Add Icons Everywhere

```tsx
import { Icon } from 'lucide-react';

<div className="flex items-center gap-2">
  <Icon className="w-4 h-4" />
  <span>Your text</span>
</div>
```

### Create Variants

```tsx
// Button variants
const buttonVariants = {
  primary: "bg-purple-500 hover:bg-purple-600",
  secondary: "bg-gray-700 hover:bg-gray-600",
  danger: "bg-red-500 hover:bg-red-600",
};

<button className={buttonVariants.primary}>
```

---

## Save Your Customizations

```bash
# Commit your changes
git add .
git commit -m "Customize dashboard with my branding"
git push
```

---

## Need Help?

- Check the component file you want to customize
- Look for similar patterns in other components
- Refer to:
  - [Tailwind CSS Docs](https://tailwindcss.com/docs)
  - [Framer Motion Docs](https://www.framer.com/motion/)
  - [Next.js Docs](https://nextjs.org/docs)
  - [Recharts Docs](https://recharts.org/)

---

**Remember**: The best portfolio projects are personalized! Make this dashboard reflect your style and the story you want to tell recruiters.
