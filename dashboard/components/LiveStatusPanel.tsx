'use client';

import { motion, useSpring, useTransform } from 'framer-motion';
import { useEffect, useState } from 'react';
import { Activity, Server, Cpu, MemoryStick } from 'lucide-react';
import { useScalingStatus } from '@/hooks/useApi';

const AnimatedCounter = ({ value, duration = 1 }: { value: number; duration?: number }) => {
  const spring = useSpring(0, { stiffness: 100, damping: 30 });
  const display = useTransform(spring, (current) => Math.round(current));
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    spring.set(value);
  }, [spring, value]);

  useEffect(() => {
    const unsubscribe = display.onChange((latest) => {
      setDisplayValue(latest);
    });
    return () => unsubscribe();
  }, [display]);

  return <span>{displayValue}</span>;
};

const CircularProgress = ({
  percentage,
  size = 120,
  strokeWidth = 8,
  color = '#3b82f6',
}: {
  percentage: number;
  size?: number;
  strokeWidth?: number;
  color?: string;
}) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (percentage / 100) * circumference;

  return (
    <svg width={size} height={size} className="transform -rotate-90">
      {/* Background circle */}
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        stroke="#27272a"
        strokeWidth={strokeWidth}
        fill="none"
      />
      {/* Progress circle */}
      <motion.circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        stroke={color}
        strokeWidth={strokeWidth}
        fill="none"
        strokeLinecap="round"
        initial={{ strokeDashoffset: circumference }}
        animate={{ strokeDashoffset: offset }}
        transition={{ duration: 1, ease: 'easeInOut' }}
        style={{
          strokeDasharray: circumference,
        }}
      />
    </svg>
  );
};

const StatusBadge = ({ status }: { status: string }) => {
  const statusConfig = {
    stable: { color: 'bg-emerald-500', text: 'Stable', pulse: false },
    scaling: { color: 'bg-amber-500', text: 'Scaling', pulse: true },
    warning: { color: 'bg-orange-500', text: 'Warning', pulse: true },
    error: { color: 'bg-red-500', text: 'Error', pulse: true },
  };

  const config = statusConfig[status as keyof typeof statusConfig] || statusConfig.stable;

  return (
    <div className="flex items-center gap-2">
      <div className="relative">
        <div className={`w-2.5 h-2.5 rounded-full ${config.color}`} />
        {config.pulse && (
          <motion.div
            className={`absolute inset-0 w-2.5 h-2.5 rounded-full ${config.color}`}
            animate={{ scale: [1, 2, 2], opacity: [1, 0, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
          />
        )}
      </div>
      <span className="text-sm font-medium text-zinc-300">{config.text}</span>
    </div>
  );
};

export default function LiveStatusPanel() {
  const { data, loading, error } = useScalingStatus();

  if (loading && !data) {
    return (
      <div className="glass rounded-2xl p-8">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-zinc-800 rounded w-1/3"></div>
          <div className="h-32 bg-zinc-800 rounded"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="glass rounded-2xl p-8">
        <div className="text-center text-red-400">
          <p className="text-sm">Failed to load status</p>
          <p className="text-xs text-zinc-500 mt-2">Make sure the API is running at localhost:8000</p>
        </div>
      </div>
    );
  }

  const cpuColor = data && data.cpu_usage > 80 ? '#ef4444' : data && data.cpu_usage > 60 ? '#f59e0b' : '#3b82f6';
  const memoryColor = data && data.memory_usage > 80 ? '#ef4444' : data && data.memory_usage > 60 ? '#f59e0b' : '#10b981';

  return (
    <motion.div
      className="glass rounded-2xl p-8"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-3">
          <div className="bg-zinc-800 rounded-xl p-3">
            <Activity className="w-6 h-6 text-blue-500" />
          </div>
          <h2 className="text-2xl font-bold">Live Status</h2>
        </div>
        {data && <StatusBadge status={data.status} />}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        {/* Replicas Counter */}
        <motion.div
          className="text-center"
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.1, duration: 0.5 }}
        >
          <div className="flex items-center justify-center mb-3">
            <Server className="w-5 h-5 text-zinc-500 mr-2" />
            <span className="text-sm text-zinc-500">Active Replicas</span>
          </div>
          <div className="text-6xl font-bold text-white">
            {data ? <AnimatedCounter value={data.current_replicas} /> : 0}
          </div>
          {data && data.desired_replicas !== data.current_replicas && (
            <motion.div
              className="text-sm text-amber-400 mt-2"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              Scaling to {data.desired_replicas}...
            </motion.div>
          )}
        </motion.div>

        {/* CPU Usage */}
        <motion.div
          className="flex flex-col items-center"
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.2, duration: 0.5 }}
        >
          <div className="flex items-center mb-3">
            <Cpu className="w-5 h-5 text-zinc-500 mr-2" />
            <span className="text-sm text-zinc-500">CPU Usage</span>
          </div>
          <div className="relative">
            <CircularProgress
              percentage={data?.cpu_usage || 0}
              color={cpuColor}
            />
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-2xl font-bold">
                {data?.cpu_usage || 0}%
              </span>
            </div>
          </div>
        </motion.div>

        {/* Memory Usage */}
        <motion.div
          className="flex flex-col items-center"
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.3, duration: 0.5 }}
        >
          <div className="flex items-center mb-3">
            <MemoryStick className="w-5 h-5 text-zinc-500 mr-2" />
            <span className="text-sm text-zinc-500">Memory Usage</span>
          </div>
          <div className="relative">
            <CircularProgress
              percentage={data?.memory_usage || 0}
              color={memoryColor}
            />
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-2xl font-bold">
                {data?.memory_usage || 0}%
              </span>
            </div>
          </div>
        </motion.div>
      </div>

      {data && data.last_scaled && (
        <motion.div
          className="mt-6 text-center text-sm text-zinc-600"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          Last scaled: {new Date(data.last_scaled).toLocaleString()}
        </motion.div>
      )}
    </motion.div>
  );
}
