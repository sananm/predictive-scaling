'use client';

import { motion } from 'framer-motion';
import { Github, Linkedin, Mail, ExternalLink } from 'lucide-react';

export default function Footer() {
  const links = [
    { icon: Github, label: 'GitHub', href: 'https://github.com/yourusername' },
    { icon: Linkedin, label: 'LinkedIn', href: 'https://linkedin.com/in/yourusername' },
    { icon: Mail, label: 'Email', href: 'mailto:your.email@example.com' },
  ];

  return (
    <footer className="relative py-12 px-6 border-t border-zinc-800">
      <div className="max-w-6xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
          {/* About */}
          <div>
            <h3 className="text-lg font-bold mb-3 text-white">
              Predictive Scaling
            </h3>
            <p className="text-sm text-zinc-500 leading-relaxed">
              An ML-powered auto-scaling system that predicts traffic patterns and scales infrastructure proactively,
              reducing costs and eliminating downtime.
            </p>
          </div>

          {/* Tech Stack */}
          <div>
            <h3 className="text-sm font-semibold mb-3 text-zinc-300">Tech Stack</h3>
            <div className="flex flex-wrap gap-2">
              {['Python', 'FastAPI', 'Prophet', 'Kubernetes', 'PostgreSQL', 'Kafka', 'React', 'Next.js'].map((tech) => (
                <span
                  key={tech}
                  className="text-xs px-2 py-1 rounded bg-zinc-800 border border-zinc-700 text-zinc-400"
                >
                  {tech}
                </span>
              ))}
            </div>
          </div>

          {/* Connect */}
          <div>
            <h3 className="text-sm font-semibold mb-3 text-zinc-300">Connect</h3>
            <div className="flex gap-3">
              {links.map((link) => (
                <motion.a
                  key={link.label}
                  href={link.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 hover:border-zinc-600 rounded-lg p-3 transition-colors"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <link.icon className="w-5 h-5 text-zinc-400" />
                </motion.a>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="pt-8 border-t border-zinc-800 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-sm text-zinc-600">
            Built with care by{' '}
            <span className="text-blue-500 font-semibold">Your Name</span>
          </p>

          <div className="flex items-center gap-4 text-sm text-zinc-600">
            <a
              href="https://github.com/yourusername/predictive-scaling"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 hover:text-blue-500 transition-colors"
            >
              View Source <ExternalLink className="w-3 h-3" />
            </a>
            <span>•</span>
            <span>{new Date().getFullYear()}</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
