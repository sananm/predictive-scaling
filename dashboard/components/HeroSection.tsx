'use client';

import { motion } from 'framer-motion';
import { Cpu, TrendingUp, Zap, Shield } from 'lucide-react';

export default function HeroSection() {
  const features = [
    {
      icon: Cpu,
      title: 'ML-Powered',
      description: 'Prophet & ensemble models predict traffic patterns',
    },
    {
      icon: TrendingUp,
      title: 'Proactive Scaling',
      description: 'Scale before the spike, not during',
    },
    {
      icon: Zap,
      title: 'Cost Optimized',
      description: 'Right-size infrastructure automatically',
    },
    {
      icon: Shield,
      title: 'Risk-Aware',
      description: 'Built-in safety checks & rollback',
    },
  ];

  return (
    <section className="relative py-20 px-6 overflow-hidden">
      <div className="relative max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-12"
        >
          <motion.h1
            className="text-6xl md:text-7xl font-bold mb-6 text-white"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.8 }}
          >
            Predictive Scaling
          </motion.h1>

          <motion.p
            className="text-xl md:text-2xl text-zinc-400 max-w-3xl mx-auto"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, duration: 0.8 }}
          >
            AI-powered infrastructure that scales{' '}
            <span className="text-blue-500 font-semibold">before</span> demand hits.
            <br />
            Reduce costs, eliminate downtime, sleep better.
          </motion.p>
        </motion.div>

        {/* Feature cards */}
        <motion.div
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6, duration: 0.8 }}
        >
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              className="glass card-hover rounded-2xl p-6"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8 + index * 0.1, duration: 0.5 }}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className="bg-zinc-800 rounded-xl p-3 w-fit mb-4">
                <feature.icon className="w-6 h-6 text-blue-500" />
              </div>
              <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
              <p className="text-zinc-500 text-sm">{feature.description}</p>
            </motion.div>
          ))}
        </motion.div>

        {/* Tech stack pills */}
        <motion.div
          className="flex flex-wrap justify-center gap-3 mt-12"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2, duration: 0.8 }}
        >
          {['Python', 'Prophet', 'FastAPI', 'Kubernetes', 'PostgreSQL', 'Kafka'].map(
            (tech, index) => (
              <motion.span
                key={tech}
                className="px-4 py-2 rounded-full bg-zinc-800/50 border border-zinc-700 text-sm text-zinc-400"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 1.4 + index * 0.05, duration: 0.3 }}
                whileHover={{ scale: 1.05, borderColor: '#3b82f6' }}
              >
                {tech}
              </motion.span>
            )
          )}
        </motion.div>
      </div>
    </section>
  );
}
