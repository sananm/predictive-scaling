'use client';

import HeroSection from '@/components/HeroSection';
import LiveStatusPanel from '@/components/LiveStatusPanel';
import PredictionsChart from '@/components/PredictionsChart';
import InteractiveDemo from '@/components/InteractiveDemo';
import ScalingTimeline from '@/components/ScalingTimeline';
import ArchitectureDiagram from '@/components/ArchitectureDiagram';
import Footer from '@/components/Footer';

export default function Home() {
  return (
    <main className="min-h-screen bg-background">
      <div className="relative z-10">
        {/* Hero Section */}
        <HeroSection />

        {/* Main Dashboard Content */}
        <div className="max-w-7xl mx-auto px-6 py-12 space-y-8">
          {/* Live Status Panel */}
          <LiveStatusPanel />

          {/* Two Column Layout */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Predictions Chart */}
            <PredictionsChart />

            {/* Interactive Demo */}
            <InteractiveDemo />
          </div>

          {/* Scaling Timeline */}
          <ScalingTimeline />

          {/* Architecture Diagram */}
          <ArchitectureDiagram />
        </div>

        {/* Footer */}
        <Footer />
      </div>
    </main>
  );
}
