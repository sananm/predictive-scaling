'use client';

import { motion } from 'framer-motion';
import { Database, Brain, GitBranch, Server, Activity, Gauge } from 'lucide-react';

const FlowArrow = ({ delay = 0 }: { delay?: number }) => (
  <motion.div
    className="flex items-center justify-center"
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    transition={{ delay, duration: 0.5 }}
  >
    <motion.div
      className="h-0.5 w-full bg-zinc-700 relative"
      initial={{ scaleX: 0 }}
      animate={{ scaleX: 1 }}
      transition={{ delay: delay + 0.2, duration: 0.8 }}
      style={{ transformOrigin: 'left' }}
    >
      <motion.div
        className="absolute right-0 top-1/2 -translate-y-1/2 w-2 h-2 bg-blue-500 rounded-full"
        animate={{
          x: [0, 10, 0],
          opacity: [1, 0.5, 1],
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
      />
    </motion.div>
  </motion.div>
);

const ComponentCard = ({
  icon: Icon,
  title,
  subtitle,
  color,
  delay = 0,
}: {
  icon: any;
  title: string;
  subtitle: string;
  color: string;
  delay?: number;
}) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay, duration: 0.5 }}
    whileHover={{ scale: 1.02, y: -2 }}
    className={`rounded-2xl p-6 bg-zinc-800/50 border border-zinc-700 hover:border-${color}-500/50 transition-colors`}
  >
    <div className="flex flex-col items-center text-center">
      <motion.div
        className="mb-4"
        whileHover={{ rotate: 360 }}
        transition={{ duration: 0.6 }}
      >
        <Icon className={`w-10 h-10 text-${color}-500`} />
      </motion.div>
      <h3 className="text-lg font-bold mb-1">{title}</h3>
      <p className="text-xs text-zinc-500">{subtitle}</p>
    </div>
  </motion.div>
);

export default function ArchitectureDiagram() {
  return (
    <motion.div
      className="glass rounded-2xl p-8"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.5 }}
    >
      <div className="flex items-center gap-3 mb-12">
        <div className="bg-zinc-800 rounded-xl p-3">
          <GitBranch className="w-6 h-6 text-blue-500" />
        </div>
        <div>
          <h2 className="text-2xl font-bold">System Architecture</h2>
          <p className="text-sm text-zinc-500 mt-1">
            End-to-end ML pipeline for intelligent scaling
          </p>
        </div>
      </div>

      {/* Desktop view - horizontal flow */}
      <div className="hidden lg:block">
        <div className="grid grid-cols-5 gap-4 items-center">
          {/* Step 1: Data Collection */}
          <ComponentCard
            icon={Database}
            title="Data Collection"
            subtitle="Metrics & Events"
            color="blue"
            delay={0.2}
          />

          <FlowArrow delay={0.4} />

          {/* Step 2: Feature Engineering */}
          <ComponentCard
            icon={Activity}
            title="Features"
            subtitle="Time & Business"
            color="cyan"
            delay={0.6}
          />

          <FlowArrow delay={0.8} />

          {/* Step 3: ML Models */}
          <ComponentCard
            icon={Brain}
            title="ML Models"
            subtitle="Prophet & Ensemble"
            color="blue"
            delay={1.0}
          />
        </div>

        <div className="my-8 flex justify-center">
          <motion.div
            initial={{ opacity: 0, scaleY: 0 }}
            animate={{ opacity: 1, scaleY: 1 }}
            transition={{ delay: 1.2, duration: 0.8 }}
            className="w-0.5 h-16 bg-zinc-700 relative"
          >
            <motion.div
              className="absolute left-1/2 -translate-x-1/2 w-2 h-2 bg-blue-500 rounded-full"
              animate={{
                y: [0, 50, 0],
                opacity: [1, 0.5, 1],
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: 'easeInOut',
              }}
            />
          </motion.div>
        </div>

        <div className="grid grid-cols-5 gap-4 items-center">
          <div className="col-start-3">
            <ComponentCard
              icon={Gauge}
              title="Decision Engine"
              subtitle="Risk & Cost Analysis"
              color="amber"
              delay={1.4}
            />
          </div>
        </div>

        <div className="my-8 flex justify-center">
          <motion.div
            initial={{ opacity: 0, scaleY: 0 }}
            animate={{ opacity: 1, scaleY: 1 }}
            transition={{ delay: 1.6, duration: 0.8 }}
            className="w-0.5 h-16 bg-zinc-700 relative"
          >
            <motion.div
              className="absolute left-1/2 -translate-x-1/2 w-2 h-2 bg-emerald-500 rounded-full"
              animate={{
                y: [0, 50, 0],
                opacity: [1, 0.5, 1],
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: 'easeInOut',
              }}
            />
          </motion.div>
        </div>

        <div className="grid grid-cols-5 gap-4 items-center">
          <div className="col-start-3">
            <ComponentCard
              icon={Server}
              title="Kubernetes"
              subtitle="Auto-scaling"
              color="emerald"
              delay={1.8}
            />
          </div>
        </div>
      </div>

      {/* Mobile view - vertical flow */}
      <div className="lg:hidden space-y-4">
        <ComponentCard
          icon={Database}
          title="Data Collection"
          subtitle="Metrics & Events"
          color="blue"
          delay={0.2}
        />

        <div className="flex justify-center">
          <motion.div
            initial={{ opacity: 0, scaleY: 0 }}
            animate={{ opacity: 1, scaleY: 1 }}
            transition={{ delay: 0.4, duration: 0.5 }}
            className="w-0.5 h-8 bg-zinc-700"
          />
        </div>

        <ComponentCard
          icon={Activity}
          title="Feature Engineering"
          subtitle="Time & Business Features"
          color="cyan"
          delay={0.6}
        />

        <div className="flex justify-center">
          <motion.div
            initial={{ opacity: 0, scaleY: 0 }}
            animate={{ opacity: 1, scaleY: 1 }}
            transition={{ delay: 0.8, duration: 0.5 }}
            className="w-0.5 h-8 bg-zinc-700"
          />
        </div>

        <ComponentCard
          icon={Brain}
          title="ML Models"
          subtitle="Prophet & Ensemble"
          color="blue"
          delay={1.0}
        />

        <div className="flex justify-center">
          <motion.div
            initial={{ opacity: 0, scaleY: 0 }}
            animate={{ opacity: 1, scaleY: 1 }}
            transition={{ delay: 1.2, duration: 0.5 }}
            className="w-0.5 h-8 bg-zinc-700"
          />
        </div>

        <ComponentCard
          icon={Gauge}
          title="Decision Engine"
          subtitle="Risk & Cost Analysis"
          color="amber"
          delay={1.4}
        />

        <div className="flex justify-center">
          <motion.div
            initial={{ opacity: 0, scaleY: 0 }}
            animate={{ opacity: 1, scaleY: 1 }}
            transition={{ delay: 1.6, duration: 0.5 }}
            className="w-0.5 h-8 bg-zinc-700"
          />
        </div>

        <ComponentCard
          icon={Server}
          title="Kubernetes"
          subtitle="Auto-scaling"
          color="emerald"
          delay={1.8}
        />
      </div>

      {/* Key features */}
      <motion.div
        className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-4"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 2.0 }}
      >
        <div className="bg-zinc-800/50 border border-zinc-700 rounded-xl p-4">
          <h4 className="text-sm font-semibold text-blue-400 mb-2">Real-time Processing</h4>
          <p className="text-xs text-zinc-500">
            Kafka streams process metrics and events in real-time for instant predictions
          </p>
        </div>
        <div className="bg-zinc-800/50 border border-zinc-700 rounded-xl p-4">
          <h4 className="text-sm font-semibold text-cyan-400 mb-2">Multi-model Ensemble</h4>
          <p className="text-xs text-zinc-500">
            Combines Prophet, LSTM, and XGBoost for robust forecasting
          </p>
        </div>
        <div className="bg-zinc-800/50 border border-zinc-700 rounded-xl p-4">
          <h4 className="text-sm font-semibold text-emerald-400 mb-2">Safe Execution</h4>
          <p className="text-xs text-zinc-500">
            Built-in verification, rollback, and audit trails for every scaling action
          </p>
        </div>
      </motion.div>
    </motion.div>
  );
}
